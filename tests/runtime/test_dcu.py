"""CPU contracts for explicit DCU selection, FWHT and the ctypes ABI boundary."""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kvserve_v1.compression import runtime
from kvserve_v1.compression.runtime import hipcomp, memory
from kvserve_v1.compression.runtime.build_hipcomp import build_command
from kvserve_v1.compression.runtime.hadamard import normalized_fwht


def config(**kwargs):
    return runtime.CompressionConfig.from_extra_config(
        {"compression": {"enable-compression-pipeline": True, **kwargs}}
    )


@pytest.mark.parametrize("codec", ["ans", "hipcomp_snappy", "hipcomp_cascaded"])
def test_explicit_codecs_do_not_change_defaults(codec):
    assert config(codec=codec).codec == codec
    assert config().codec == "ans"
    assert runtime.API_VERSION == 1


@pytest.mark.parametrize("codec", ["snappy", "ANS", "hipcomp", "auto", ""])
def test_no_implicit_codec_remapping(codec):
    with pytest.raises(ValueError, match="Unsupported compression codec"):
        config(codec=codec)


@pytest.mark.parametrize("chunk", [0, 255, 257, 16777217, -256, True, "16384", 16384.0])
def test_invalid_chunk_bytes(chunk):
    with pytest.raises(ValueError, match="hipcomp_chunk_bytes"):
        config(codec="hipcomp_snappy", hipcomp_chunk_bytes=chunk)


def test_raw_transport_ignores_codec_configuration():
    raw = runtime.CompressionConfig.from_extra_config(
        {"raw_aggregate": True, "compression": {"codec": "unknown"}}
    )
    assert raw.uses_aggregate and not raw.uses_codec


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [1, 2, 8, 256])
def test_fwht_matches_normalized_matrix_and_inverse(dtype, size):
    torch.manual_seed(17)
    matrix = torch.ones(1, 1)
    while len(matrix) < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    matrix /= size**0.5
    source = torch.randn(2, 3, size).to(dtype)
    expected = source.float() @ matrix
    scratch, encoded, decoded = (
        source.clone(),
        torch.empty_like(source),
        torch.empty_like(source),
    )
    normalized_fwht(scratch, encoded, size)
    atol = 0.04 if dtype == torch.bfloat16 else 0.004
    torch.testing.assert_close(encoded.float(), expected, atol=atol, rtol=0.02)
    normalized_fwht(encoded.clone(), decoded, size)
    torch.testing.assert_close(decoded.float(), source.float(), atol=atol, rtol=0.02)


def test_fwht_main_and_worker_threads_have_same_result():
    source = torch.randn(3, 256).bfloat16()

    def transform():
        output = torch.empty_like(source)
        normalized_fwht(source.clone(), output, 256)
        return output

    expected = transform()
    with ThreadPoolExecutor(2) as executor:
        for output in executor.map(lambda _: transform(), range(4)):
            assert torch.equal(output, expected)


def test_fwht_rejects_alias_and_invalid_shape():
    source = torch.ones(2, 8)
    with pytest.raises(ValueError, match="overlap"):
        normalized_fwht(source, source, 8)
    with pytest.raises(ValueError, match="power of two"):
        normalized_fwht(source, torch.empty_like(source), 3)
    with pytest.raises(ValueError, match="contiguous"):
        normalized_fwht(source.t(), torch.empty_like(source.t()), 8)
    with pytest.raises(ValueError, match="shapes"):
        normalized_fwht(source, torch.empty(8), 8)


def fake_library(*, abi=1, mask=6, status=0):
    library = SimpleNamespace(
        kvs_hipcomp_abi_version=Mock(return_value=abi),
        kvs_hipcomp_algorithm_mask=Mock(return_value=mask),
        kvs_hipcomp_compress=Mock(),
        kvs_hipcomp_decompress=Mock(),
    )

    def plan(algorithm, size, chunk, workspace, output):
        ct.cast(workspace, ct.POINTER(ct.c_uint64))[0] = 4096
        ct.cast(output, ct.POINTER(ct.c_uint64))[0] = size + 1024
        return status

    library.kvs_hipcomp_plan = Mock(side_effect=plan)
    return library


def test_native_binding_and_explicit_algorithm(monkeypatch):
    lib = fake_library()
    monkeypatch.setattr(hipcomp.ct, "CDLL", Mock(return_value=lib))
    native = hipcomp.NativeHipcomp("test.so")
    assert native.plan("hipcomp_cascaded", 16385, 16384) == hipcomp.HipcompPlan(
        4096, 17409
    )
    assert lib.kvs_hipcomp_plan.call_args.args[:3] == (2, 16385, 16384)
    assert len(lib.kvs_hipcomp_compress.argtypes) == 11
    assert len(lib.kvs_hipcomp_decompress.argtypes) == 11


@pytest.mark.parametrize(
    "abi,mask,status,message",
    [
        (0, 6, 0, "ABI mismatch"),
        (1, 2, 0, "does not implement"),
        (1, 6, 2001, "status=2001"),
    ],
)
def test_native_contract_errors(monkeypatch, abi, mask, status, message):
    monkeypatch.setattr(
        hipcomp.ct,
        "CDLL",
        Mock(return_value=fake_library(abi=abi, mask=mask, status=status)),
    )
    with pytest.raises(RuntimeError, match=message):
        hipcomp.NativeHipcomp("test.so").plan("hipcomp_cascaded", 1024, 16384)


def test_missing_library_is_actionable_and_not_a_raw_fallback(monkeypatch):
    monkeypatch.setattr(hipcomp.ct, "CDLL", Mock(side_effect=OSError("not found")))
    with pytest.raises(RuntimeError, match="not ABI-compatible"):
        hipcomp.NativeHipcomp("missing.so")


def test_old_library_symbols_are_rejected(monkeypatch):
    monkeypatch.setattr(hipcomp.ct, "CDLL", Mock(return_value=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="Not a KVServe runtime HIP ABI"):
        hipcomp.NativeHipcomp("old.so")


def test_explicit_library_path_is_used(monkeypatch, tmp_path):
    loader = Mock()
    path = tmp_path / "my_hip.so"
    monkeypatch.setenv("KVSERVE_HIPCOMP_LIBRARY", str(path))
    monkeypatch.setattr(hipcomp, "_load_library", loader)
    hipcomp.native_library()
    loader.assert_called_once_with(str(path.resolve()))


def test_wrong_platform_and_cuda_cpp_backend_fail_early(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="HIP/DCU"):
        hipcomp.validate_hip_backend(torch.device("cuda", 0))
    monkeypatch.setattr(torch.version, "hip", "6.3.test")
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "cpp")
    with pytest.raises(ValueError, match="CUDA-only"):
        hipcomp.validate_hip_backend(torch.device("cuda", 0))


def test_dcu_plan_budgets_every_slot_and_dispatches_without_nvcomp(monkeypatch):
    probe = Mock(return_value=(4096, (65536, 65536), 512))
    monkeypatch.setattr(memory, "_probe_hipcomp", probe)
    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    specs = {"layer": runtime.CacheSpec(0, 64, 2, 256, torch.bfloat16)}
    single = memory.build_memory_plan(
        config(codec="hipcomp_cascaded", slot_count=1),
        specs,
        device=torch.device("cuda", 0),
    )
    multi = memory.build_memory_plan(
        config(codec="hipcomp_cascaded", slot_count=3),
        specs,
        device=torch.device("cuda", 0),
    )
    assert single.codec_workspace_bytes == 4096
    assert multi.codec_workspace_bytes == 3 * 4096
    assert multi.retained_driver_bytes == 3 * 512
    assert multi.layout.arena_bytes >= 65536
    assert multi.reserved_bytes > single.reserved_bytes
    assert probe.call_args.args[-2:] == ("hipcomp_cascaded", 16384)


def test_dcu_manager_owns_distinct_slot_workspaces_without_nvcomp_allocator(
    monkeypatch,
):
    from kvserve_v1.compression.runtime import manager as manager_module

    monkeypatch.setenv("KVSERVE_COMPRESSION_BACKEND", "python")
    monkeypatch.setattr(
        memory, "_probe_hipcomp", Mock(return_value=(4096, (65536, 65536), 0))
    )
    specs = {"layer": runtime.CacheSpec(0, 64, 2, 256, torch.bfloat16)}
    device = torch.device("cuda", 0)
    plan = memory.build_memory_plan(
        config(codec="hipcomp_snappy", slot_count=3), specs, device=device
    )
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    streams = iter(
        SimpleNamespace(
            cuda_stream=i, device=device, wait_event=Mock(), synchronize=Mock()
        )
        for i in range(3)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: next(streams))
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    register = Mock(
        side_effect=AssertionError("DCU must not register nvCOMP allocator")
    )
    monkeypatch.setattr(manager_module, "register_workspace_allocators", register)
    codec_factory = Mock(side_effect=lambda *args: SimpleNamespace())
    monkeypatch.setattr(hipcomp, "HIPCOMPCodec", codec_factory)
    manager = runtime.CompressionRuntime(plan, "HND", 1, device)
    assert codec_factory.call_count == 3
    buffers = [call.args[2] for call in codec_factory.call_args_list]
    assert len({buffer.data_ptr() for buffer in buffers}) == 3
    assert all(buffer.numel() == 4096 for buffer in buffers)
    assert all(
        call.args[-2:] == ("hipcomp_snappy", 16384)
        for call in codec_factory.call_args_list
    )
    assert not manager._owns_nvcomp_allocator
    register.assert_not_called()
    manager.shutdown()


def test_build_command_uses_explicit_toolchain_prefix_and_arch(tmp_path):
    args = SimpleNamespace(
        hipcomp_root=str(tmp_path),
        include_dir=[],
        library_dir=[],
        hipcc="/opt/dtk/bin/hipcc",
        hipcomp_library="hipcomp",
        output=tmp_path / "runtime.so",
        offload_arch="gfx906",
    )
    command = build_command(args)
    assert command[0] == "/opt/dtk/bin/hipcc"
    assert "-x" in command and "hip" in command and "-lhipcomp" in command
    assert "--offload-arch=gfx906" in command
    assert any(str(tmp_path / "include") in arg for arg in command)


def test_default_profiles_are_valid_and_use_two_slots():
    import json

    root = Path(__file__).resolve().parents[2]
    for codec in ("snappy", "cascaded"):
        profile = json.loads((root / f"configs/runtime/dcu_{codec}.json").read_text())
        parsed = runtime.CompressionConfig.from_extra_config(profile)
        assert parsed.codec == f"hipcomp_{codec}"
        assert parsed.slot_count == 2 and parsed.logical_blocks_per_chunk == 2


def test_hip_error_waits_for_stream_before_releasing_buffers(monkeypatch):
    codec = object.__new__(hipcomp.HIPCOMPCodec)
    codec.stream = SimpleNamespace(device=torch.device("cuda", 1), synchronize=Mock())
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    operation = Mock(return_value=4)
    operation.__name__ = "decode"
    with pytest.raises(RuntimeError, match="malformed frame"):
        codec._run(operation)
    codec.stream.synchronize.assert_called_once()


def test_codec_calls_native_with_fixed_workspace_stream_and_real_payload_length(
    monkeypatch,
):
    device = torch.device("cuda", 1)
    stream = SimpleNamespace(device=device, cuda_stream=12345, synchronize=Mock())

    def buffer(address, size):
        return SimpleNamespace(
            device=device,
            dtype=torch.uint8,
            ndim=1,
            is_contiguous=lambda: True,
            numel=lambda: size,
            data_ptr=lambda: address,
        )

    workspace, source, output = (
        buffer(0x30000, 4096),
        buffer(0x10000, 1024),
        buffer(0x20000, 2048),
    )

    def compress(*args):
        ct.cast(args[-1], ct.POINTER(ct.c_uint64))[0] = 777
        return 0

    encoder, decoder = Mock(side_effect=compress), Mock(return_value=0)
    encoder.__name__, decoder.__name__ = "compress", "decompress"
    native = SimpleNamespace(
        plan=Mock(return_value=hipcomp.HipcompPlan(4096, 2048)),
        lib=SimpleNamespace(
            kvs_hipcomp_compress=encoder, kvs_hipcomp_decompress=decoder
        ),
    )
    monkeypatch.setattr(hipcomp, "native_library", lambda: native)
    monkeypatch.setattr(hipcomp, "validate_hip_backend", Mock())
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    codec = hipcomp.HIPCOMPCodec(
        stream, (1024, 512), workspace, "hipcomp_cascaded", 16384
    )
    event = SimpleNamespace(record=Mock())
    encoded = codec.encode(source, output, 512, event)
    assert encoded.codec_bytes == 777 and encoded.u8_bytes == 512
    assert encoder.call_args.args[:-1] == (
        2,
        0x10000,
        512,
        1024,
        16384,
        0x20000,
        2048,
        0x30000,
        4096,
        12345,
    )
    assert codec.decode(output, source, 777, 512, event) is event
    assert decoder.call_args.args == (
        2,
        0x20000,
        777,
        512,
        1024,
        16384,
        0x10000,
        1024,
        0x30000,
        4096,
        12345,
    )
    assert event.record.call_count == 2
    with pytest.raises(ValueError, match="planned input"):
        codec.encode(source, output, 1025, event)
    with pytest.raises(ValueError, match="memory plan"):
        codec.decode(output, source, 2049, 512, event)
