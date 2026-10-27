# SPDX-License-Identifier: Apache-2.0
"""Normalized FWHT on the caller's stream, without a Humming/CUDA-only import.

Promotes the runtime's existing worker-thread butterfly implementation to all
threads. It computes the same normalized transform as the DCU snapshot's
fast_hadamard_transform path; signs remain managed by CompressionRuntime.
Both supplied buffers are already charged to the runtime memory plan.
"""

from __future__ import annotations

import math

import torch


def normalized_fwht(
    inputs: torch.Tensor, outputs: torch.Tensor, block_size: int
) -> None:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    if inputs.shape != outputs.shape or inputs.numel() % block_size:
        raise ValueError("Hadamard input/output shapes do not match block_size")
    if inputs.device != outputs.device or inputs.dtype != outputs.dtype:
        raise ValueError("Hadamard buffers must share dtype and device")
    if not inputs.is_contiguous() or not outputs.is_contiguous():
        raise ValueError("Hadamard buffers must be contiguous")
    if inputs.numel() == 0:
        return
    start_a, start_b = inputs.data_ptr(), outputs.data_ptr()
    nbytes = inputs.numel() * inputs.element_size()
    if max(start_a, start_b) < min(start_a + nbytes, start_b + nbytes):
        raise ValueError("Hadamard scratch buffers must not overlap")
    src = inputs.view(-1, block_size)
    output_flat = outputs.view(-1, block_size)
    dst = output_flat
    step = 1
    while step < block_size:
        src_pairs = src.view(-1, block_size // (2 * step), 2, step)
        dst_pairs = dst.view(-1, block_size // (2 * step), 2, step)
        even, odd = src_pairs[:, :, 0, :], src_pairs[:, :, 1, :]
        torch.add(even, odd, out=dst_pairs[:, :, 0, :])
        torch.sub(even, odd, out=dst_pairs[:, :, 1, :])
        src, dst = dst, src
        step <<= 1
    if src.data_ptr() != output_flat.data_ptr():
        output_flat.copy_(src)
    output_flat.mul_(1.0 / math.sqrt(block_size))
