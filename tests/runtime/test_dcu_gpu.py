"""Real HIP tests: python -m pytest tests/runtime/test_dcu_gpu.py -q.

On HIP machines a missing/broken native library is a failure, not a skip.
These tests do not replace cross-node SGLang/Mooncake/Store integration tests.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kvserve_v1.compression.runtime import (
    CacheSpec,
    CompressionConfig,
    CompressionRuntime,
    build_memory_plan,
)
from kvserve_v1.compression.runtime.hipcomp import HIPCOMPCodec, native_library

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="Requires actual HIP/DCU",
)
CODECS = ["hipcomp_snappy", "hipcomp_cascaded"]


@pytest.mark.parametrize("codec_name", CODECS)
@pytest.mark.parametrize("length", [1, 16383, 16384, 16385, 65537])
def test_real_lossless_codec(codec_name, length):
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    bounds = native_library().plan(codec_name, length, 16384)
    with torch.cuda.stream(stream):
        workspace = torch.empty(
            bounds.workspace_bytes, dtype=torch.uint8, device=device
        )
        source = torch.randint(0, 256, (length,), dtype=torch.uint8, device=device)
        output = torch.empty(bounds.max_output_bytes, dtype=torch.uint8, device=device)
        decoded = torch.empty_like(source)
        codec = HIPCOMPCodec(stream, (length,), workspace, codec_name, 16384)
        event = torch.cuda.Event()
        try:
            encoded = codec.encode(source, output, length, event)
            codec.decode(
                output, decoded, encoded.codec_bytes, length, event
            ).synchronize()
            assert torch.equal(source, decoded)
            output[0].fill_(0)  # Corrupt the magic, must not silently copy bytes.
            with pytest.raises(RuntimeError, match="status=4"):
                codec.decode(output, decoded, encoded.codec_bytes, length, event)
        finally:
            stream.synchronize()


@pytest.mark.parametrize("codec_name", CODECS)
def test_two_workspaces_from_concurrent_worker_threads(codec_name):
    device = torch.device("cuda", torch.cuda.current_device())

    def roundtrip(seed):
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            bounds = native_library().plan(codec_name, 32769, 16384)
            with torch.cuda.stream(stream):
                workspace = torch.empty(
                    bounds.workspace_bytes, dtype=torch.uint8, device=device
                )
                source = torch.full((32769,), seed, dtype=torch.uint8, device=device)
                output = torch.empty(
                    bounds.max_output_bytes, dtype=torch.uint8, device=device
                )
                decoded = torch.empty_like(source)
                codec = HIPCOMPCodec(stream, (32769,), workspace, codec_name, 16384)
                event = torch.cuda.Event()
                try:
                    encoded = codec.encode(source, output, source.numel(), event)
                    codec.decode(
                        output, decoded, encoded.codec_bytes, source.numel(), event
                    ).synchronize()
                    assert torch.equal(source, decoded)
                finally:
                    stream.synchronize()

    with ThreadPoolExecutor(2) as executor:
        list(executor.map(roundtrip, (17, 31)))


@pytest.mark.parametrize("codec_name", CODECS)
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("lossless", [False, True])
def test_spark_geometry_partial_pages_and_cross_slot_restore(
    codec_name, tp_rank, lossless
):
    """Nine FA layers, local KV heads=2, head_dim=256, TP=2, BF16, 63/64/65 tokens."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = CompressionConfig.from_extra_config(
        {
            "compression": {
                "enable-compression-pipeline": True,
                "codec": codec_name,
                "logical_blocks_per_chunk": 2,
                "slot_count": 2,
                **({"pipeline": ["codec"]} if lossless else {}),
            }
        }
    )
    specs = {
        f"layers.{i}": CacheSpec(i, 64, 2, 256, torch.bfloat16) for i in range(3, 36, 4)
    }
    plan = build_memory_plan(
        config, specs, num_layers=36, tp_size=2, tp_rank=tp_rank, device=device
    )
    managers = []
    try:
        for is_producer in (True, False):
            manager = CompressionRuntime(plan, "HND", 1, device)
            managers.append(manager)
            caches = {
                name: (
                    torch.randn(4, 2, 64, 512, device=device, dtype=torch.bfloat16)
                    if is_producer
                    else torch.full(
                        (4, 2, 64, 512), -9, device=device, dtype=torch.bfloat16
                    )
                )
                for name in specs
            }
            manager.bind_kv_caches(caches, dict.fromkeys(specs, 0))
        producer, consumer = managers
        assert producer.wire_profile == consumer.wire_profile
        for tokens in (63, 64, 65):
            count = (tokens + 63) // 64
            for source_slot in range(2):
                destination_slot = 1 - source_slot
                for part in (0, 1):
                    encoded = producer.encode(
                        source_slot,
                        [[2, 0][:count]],
                        part,
                        valid_tokens_by_group={0: tokens},
                    )
                    encoded.event.synchronize()
                    src, dst = (
                        producer.slots[source_slot],
                        consumer.slots[destination_slot],
                    )
                    dst.arena_a[: encoded.codec_bytes].copy_(
                        src.arena_a[: encoded.codec_bytes]
                    )
                    aux_bytes = producer.aux_bytes(part, {0: tokens})
                    dst.aux[:aux_bytes].copy_(src.aux[:aux_bytes])
                    dst.raw_tail[: encoded.raw_tail_bytes].copy_(
                        src.raw_tail[: encoded.raw_tail_bytes]
                    )
                    torch.cuda.synchronize(device)
                    consumer.decode(
                        destination_slot,
                        [[1, 3][:count]],
                        part,
                        encoded.codec_bytes,
                        encoded.u8_bytes,
                        valid_tokens_by_group={0: tokens},
                    ).synchronize()
                    section = slice(part * 256, (part + 1) * 256)
                    for name in specs:
                        expected = (
                            producer.pipelines[0]
                            .kv_caches[name][[2, 0][:count], :, :, section]
                            .permute(0, 2, 1, 3)
                            .flatten(0, 1)[:tokens]
                        )
                        actual = (
                            consumer.pipelines[0]
                            .kv_caches[name][[1, 3][:count], :, :, section]
                            .permute(0, 2, 1, 3)
                            .flatten(0, 1)[:tokens]
                        )
                        if lossless:
                            assert torch.equal(actual, expected)
                        else:
                            assert torch.isfinite(actual).all()
                            relative_l2 = torch.linalg.vector_norm(
                                actual.float() - expected.float()
                            ) / torch.linalg.vector_norm(expected.float())
                            assert relative_l2 < 0.5
                        tail = tokens // 64 * 64
                        assert torch.equal(actual[tail:], expected[tail:])
    finally:
        for manager in reversed(managers):
            manager.shutdown()
