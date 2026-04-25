# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable sparse MLA Triton kernels."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _merge_two_subsets_with_sink_kernel(
    out0_ptr,
    lse0_ptr,
    out1_ptr,
    lse1_ptr,
    sink_ptr,
    output_ptr,
    stride_out0_t: tl.constexpr,
    stride_out0_h: tl.constexpr,
    stride_out0_d: tl.constexpr,
    stride_lse0_t: tl.constexpr,
    stride_lse0_h: tl.constexpr,
    stride_out1_t: tl.constexpr,
    stride_out1_h: tl.constexpr,
    stride_out1_d: tl.constexpr,
    stride_lse1_t: tl.constexpr,
    stride_lse1_h: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < head_dim

    lse0 = tl.load(lse0_ptr + token_idx * stride_lse0_t + head_idx * stride_lse0_h)
    lse1 = tl.load(lse1_ptr + token_idx * stride_lse1_t + head_idx * stride_lse1_h)
    sink = tl.load(sink_ptr + head_idx)
    merge_max = tl.maximum(tl.maximum(lse0, lse1), sink)

    weight0 = tl.exp(lse0 - merge_max)
    weight1 = tl.exp(lse1 - merge_max)
    weight_sink = tl.exp(sink - merge_max)
    denom = weight0 + weight1 + weight_sink

    out0 = tl.load(
        out0_ptr
        + token_idx * stride_out0_t
        + head_idx * stride_out0_h
        + offsets * stride_out0_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out1 = tl.load(
        out1_ptr
        + token_idx * stride_out1_t
        + head_idx * stride_out1_h
        + offsets * stride_out1_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    merged = (out0 * weight0 + out1 * weight1) / denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        merged,
        mask=mask,
    )


def merge_two_sparse_mla_subsets_with_sink(
    subset0_output: torch.Tensor,
    subset0_lse: torch.Tensor,
    subset1_output: torch.Tensor,
    subset1_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert subset0_output.shape == subset1_output.shape
    assert subset0_output.shape == output.shape
    assert subset0_lse.shape == subset1_lse.shape
    assert subset0_lse.shape == subset0_output.shape[:2]
    assert attn_sink.shape[0] == subset0_output.shape[1]
    assert subset0_output.is_cuda
    assert subset1_output.is_cuda
    assert output.is_cuda

    num_tokens, num_heads, head_dim = subset0_output.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _merge_two_subsets_with_sink_kernel[grid](
        subset0_output,
        subset0_lse,
        subset1_output,
        subset1_lse,
        attn_sink,
        output,
        subset0_output.stride(0),
        subset0_output.stride(1),
        subset0_output.stride(2),
        subset0_lse.stride(0),
        subset0_lse.stride(1),
        subset1_output.stride(0),
        subset1_output.stride(1),
        subset1_output.stride(2),
        subset1_lse.stride(0),
        subset1_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
