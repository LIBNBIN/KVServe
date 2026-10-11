"""Versioned fixed-profile KV compression API for transfer/storage adapters.

The caller owns cache registration and slot scheduling. The runtime owns
bounded codec buffers and returns CUDA events; callers must fence them before
network reads or slot reuse. No vLLM import or implicit network operation occurs.
"""

from .config import CompressionConfig
from .manager import CompressionRuntime
from .memory import build_memory_plan
from .spec import CacheSpec

API_VERSION = 1
__all__ = [
    "API_VERSION",
    "CacheSpec",
    "CompressionConfig",
    "CompressionRuntime",
    "build_memory_plan",
]
