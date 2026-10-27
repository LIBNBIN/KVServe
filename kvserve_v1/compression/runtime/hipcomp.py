# SPDX-License-Identifier: Apache-2.0
"""Explicit hipCOMP backend for runtime API 1 (not an nvidia namespace shim).

The C ABI owns no device allocations. Each codec uses its runtime slot's fixed
workspace and the caller's HIP stream. Errors never fall back to raw copies.
"""

from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from .codec import EncodedChunk

ABI_VERSION = 1
ALGORITHMS = {"hipcomp_snappy": 1, "hipcomp_cascaded": 2}
DEFAULT_LIBRARY = (
    Path(__file__).resolve().parent / "_native/libkvserve_hipcomp_runtime.so"
)
_ERRORS = {
    1: "invalid buffer, capacity, alignment or argument",
    2: "size overflow or unsupported size",
    3: "compressed output exceeds its bound",
    4: "malformed frame or algorithm/chunk/length mismatch",
    5: "decompressed length does not match the expected length",
    6: "native host allocation or unexpected C++ exception",
}


def _check_status(status: int, operation: str) -> None:
    if status:
        detail = _ERRORS.get(
            status, "HIP/hipCOMP error (1000+HIP, 2000+hipCOMP, 3000+chunk status)"
        )
        raise RuntimeError(f"hipCOMP {operation} failed: status={status}: {detail}")


@dataclass(frozen=True)
class HipcompPlan:
    workspace_bytes: int
    max_output_bytes: int


class NativeHipcomp:
    def __init__(self, path: str):
        try:
            self.lib = ct.CDLL(path)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot load KVServe HIP backend {path!r}. Build it with "
                "scripts/build_hipcomp_runtime.py using the DCU toolchain and "
                "hipCOMP headers/library; set KVSERVE_HIPCOMP_LIBRARY to its .so "
                "and check LD_LIBRARY_PATH/ldd for dependencies. The old "
                "libkvserve_hipcomp_snappy.so is not ABI-compatible."
            ) from exc
        u32, u64, ptr = ct.c_uint32, ct.c_uint64, ct.c_void_p
        signatures = {
            "kvs_hipcomp_abi_version": (u32, []),
            "kvs_hipcomp_algorithm_mask": (u32, []),
            "kvs_hipcomp_plan": (
                ct.c_int,
                [u32, u64, u64, ct.POINTER(u64), ct.POINTER(u64)],
            ),
            "kvs_hipcomp_compress": (
                ct.c_int,
                [u32, ptr, u64, u64, u64, ptr, u64, ptr, u64, u64, ct.POINTER(u64)],
            ),
            "kvs_hipcomp_decompress": (
                ct.c_int,
                [u32, ptr, u64, u64, u64, u64, ptr, u64, ptr, u64, u64],
            ),
        }
        try:
            for name, (restype, argtypes) in signatures.items():
                function = getattr(self.lib, name)
                function.restype, function.argtypes = restype, argtypes
        except AttributeError as exc:
            raise RuntimeError(
                f"Not a KVServe runtime HIP ABI {ABI_VERSION} library: {path}"
            ) from exc
        abi = self.lib.kvs_hipcomp_abi_version()
        if abi != ABI_VERSION:
            raise RuntimeError(
                f"KVServe HIP ABI mismatch: expected {ABI_VERSION}, got {abi}"
            )
        self.algorithm_mask = self.lib.kvs_hipcomp_algorithm_mask()

    def plan(self, codec: str, max_input: int, chunk_bytes: int) -> HipcompPlan:
        algorithm = ALGORITHMS[codec]
        if not self.algorithm_mask & (1 << algorithm):
            raise RuntimeError(f"Loaded HIP library does not implement {codec}")
        if max_input <= 0:
            raise ValueError("hipCOMP max_input must be positive")
        workspace, output = ct.c_uint64(), ct.c_uint64()
        _check_status(
            self.lib.kvs_hipcomp_plan(
                algorithm, max_input, chunk_bytes, ct.byref(workspace), ct.byref(output)
            ),
            "plan",
        )
        if workspace.value == 0 or workspace.value % 256 or output.value == 0:
            raise RuntimeError("Invalid memory bounds returned by HIP library")
        return HipcompPlan(workspace.value, output.value)


@lru_cache(maxsize=8)
def _load_library(path: str) -> NativeHipcomp:
    return NativeHipcomp(path)


def native_library() -> NativeHipcomp:
    return _load_library(
        str(Path(os.environ.get("KVSERVE_HIPCOMP_LIBRARY", DEFAULT_LIBRARY)).resolve())
    )


def validate_hip_backend(device: torch.device) -> None:
    from .cpp_backend import enabled as cpp_enabled

    if device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "hipcomp_* codecs require a HIP/DCU PyTorch build and a cuda-type HIP device"
        )
    if cpp_enabled():
        raise ValueError(
            "DCU runtime requires KVSERVE_COMPRESSION_BACKEND=python; the cpp backend is CUDA-only"
        )


class HIPCOMPCodec:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        u8_sizes: tuple[int, ...],
        workspace: torch.Tensor,
        codec: str,
        chunk_bytes: int,
    ):
        validate_hip_backend(stream.device)
        self.stream, self.workspace = stream, workspace
        self.algorithm, self.chunk_bytes = ALGORITHMS[codec], chunk_bytes
        self.max_input = max(u8_sizes)
        self.native = native_library()
        with torch.cuda.device(stream.device):
            self.bounds = self.native.plan(codec, self.max_input, chunk_bytes)
        self._validate_buffer(workspace, self.bounds.workspace_bytes)

    def _validate_buffer(self, tensor: torch.Tensor, minimum: int) -> None:
        if (
            tensor.device != self.stream.device
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.numel() < minimum
            or tensor.data_ptr() % 256
        ):
            raise ValueError(
                "hipCOMP requires aligned contiguous 1-D uint8 buffers on the codec's device with sufficient capacity"
            )

    def _run(self, function, *args) -> None:
        with torch.cuda.device(self.stream.device):
            try:
                _check_status(function(*args), function.__name__)
            except BaseException:
                # A failed library call may have already submitted device work.
                self.stream.synchronize()
                raise

    def encode(self, source, output, u8_bytes, event) -> EncodedChunk:
        if not 0 < u8_bytes <= self.max_input:
            raise ValueError("hipCOMP encode length exceeds the planned input")
        self._validate_buffer(source, u8_bytes)
        self._validate_buffer(output, self.bounds.max_output_bytes)
        written = ct.c_uint64()
        self._run(
            self.native.lib.kvs_hipcomp_compress,
            self.algorithm,
            source.data_ptr(),
            u8_bytes,
            self.max_input,
            self.chunk_bytes,
            output.data_ptr(),
            output.numel(),
            self.workspace.data_ptr(),
            self.workspace.numel(),
            self.stream.cuda_stream,
            ct.byref(written),
        )
        if not 0 < written.value <= self.bounds.max_output_bytes:
            raise RuntimeError("HIP library returned an invalid compressed length")
        event.record(self.stream)
        return EncodedChunk(output, event, u8_bytes, payload_bytes=written.value)

    def decode(self, source, output, codec_bytes, u8_bytes, event):
        if (
            not 0 < u8_bytes <= self.max_input
            or not 0 < codec_bytes <= self.bounds.max_output_bytes
        ):
            raise ValueError("hipCOMP decode lengths exceed the memory plan")
        self._validate_buffer(source, codec_bytes)
        self._validate_buffer(output, u8_bytes)
        self._run(
            self.native.lib.kvs_hipcomp_decompress,
            self.algorithm,
            source.data_ptr(),
            codec_bytes,
            u8_bytes,
            self.max_input,
            self.chunk_bytes,
            output.data_ptr(),
            output.numel(),
            self.workspace.data_ptr(),
            self.workspace.numel(),
            self.stream.cuda_stream,
        )
        event.record(self.stream)
        return event
