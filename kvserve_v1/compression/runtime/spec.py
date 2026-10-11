"""Cache geometry accepted by the fixed-profile runtime, independent of vLLM."""

from dataclasses import dataclass

import torch


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CacheSpec:
    layer_index: int
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    head_size_v: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
