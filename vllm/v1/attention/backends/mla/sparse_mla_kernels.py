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


@triton.jit
def _merge_single_subset_with_sink_kernel(
    subset_output_ptr,
    subset_lse_ptr,
    sink_ptr,
    output_ptr,
    stride_subset_t: tl.constexpr,
    stride_subset_h: tl.constexpr,
    stride_subset_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
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

    subset_lse = tl.load(
        subset_lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h
    )
    sink = tl.load(sink_ptr + head_idx)
    merge_max = tl.maximum(subset_lse, sink)

    subset_weight = tl.exp(subset_lse - merge_max)
    sink_weight = tl.exp(sink - merge_max)
    denom = subset_weight + sink_weight
    subset_output = tl.load(
        subset_output_ptr
        + token_idx * stride_subset_t
        + head_idx * stride_subset_h
        + offsets * stride_subset_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    merged = subset_output * subset_weight / denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        merged,
        mask=mask,
    )


def merge_sparse_mla_subset_with_sink(
    subset_output: torch.Tensor,
    subset_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert subset_output.shape == output.shape
    assert subset_lse.shape == subset_output.shape[:2]
    assert attn_sink.shape[0] == subset_output.shape[1]
    assert subset_output.is_cuda
    assert subset_lse.is_cuda
    assert attn_sink.is_cuda
    assert output.is_cuda

    num_tokens, num_heads, head_dim = subset_output.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _merge_single_subset_with_sink_kernel[grid](
        subset_output,
        subset_lse,
        attn_sink,
        output,
        subset_output.stride(0),
        subset_output.stride(1),
        subset_output.stride(2),
        subset_lse.stride(0),
        subset_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


@triton.jit
def _accumulate_gathered_attention_chunk_kernel(
    q_ptr,
    kv_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    HAS_SLOT_IDS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_idx * stride_q_h
        + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    valid_len = tl.load(lens_ptr + token_idx)

    for candidate_idx in range(0, num_candidates):
        is_valid = (candidate_offset + candidate_idx) < valid_len
        if HAS_SLOT_IDS:
            slot_id = tl.load(
                slot_ids_ptr
                + token_idx * stride_slot_t
                + candidate_idx * stride_slot_c
            )
            is_valid = is_valid & (slot_id >= 0)

        if is_valid:
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_gathered_sparse_mla_attention_chunk(
    q: torch.Tensor,
    kv: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
    slot_ids: torch.Tensor | None = None,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"Expected kv shape [T, K, D], got {kv.shape}"
    assert q.shape[0] == kv.shape[0]
    assert q.shape[-1] == kv.shape[-1]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape == q.shape[:2]
    assert denom.shape == q.shape[:2]
    assert acc.shape == q.shape
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert q.is_cuda and kv.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    if slot_ids is not None:
        if slot_ids.dim() == 3:
            assert slot_ids.shape[1] == 1
            slot_ids = slot_ids[:, 0]
        assert slot_ids.dim() == 2
        assert slot_ids.shape == kv.shape[:2]
        assert slot_ids.is_cuda

    num_tokens, num_heads, head_dim = q.shape
    num_candidates = kv.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_gathered_attention_chunk_kernel[grid](
        q,
        kv,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        slot_ids.stride(0) if slot_ids is not None else 0,
        slot_ids.stride(1) if slot_ids is not None else 0,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        HAS_SLOT_IDS=slot_ids is not None,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_global_slots_attention_chunk_kernel(
    q_ptr,
    k_cache_ptr,
    slot_ids_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_slot_t: tl.constexpr,
    stride_slot_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_idx * stride_q_h
        + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    valid_len = tl.load(lens_ptr + token_idx)

    fp8_mask = offsets < fp8_dim
    rope_mask = (offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        slot_id = tl.load(
            slot_ids_ptr
            + token_idx * stride_slot_t
            + candidate_idx * stride_slot_c
        )
        is_valid = ((candidate_offset + candidate_idx) < valid_len) & (slot_id >= 0)

        if is_valid:
            block_idx = slot_id // cache_block_size
            pos_in_block = slot_id % cache_block_size
            cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    lens: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int = 0,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if slot_ids.dim() == 3:
        assert slot_ids.shape[1] == 1
        slot_ids = slot_ids[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert slot_ids.dim() == 2
    assert slot_ids.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert max_score.shape == q.shape[:2]
    assert denom.shape == q.shape[:2]
    assert acc.shape == q.shape
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda and slot_ids.is_cuda and lens.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, num_heads, head_dim = q.shape
    num_candidates = slot_ids.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_fp8ds_global_slots_attention_chunk_kernel[grid](
        q,
        k_cache,
        slot_ids,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        slot_ids.stride(0),
        slot_ids.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _accumulate_fp8ds_paged_attention_chunk_kernel(
    q_ptr,
    k_cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_block_table_t,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_idx * stride_q_h
        + offsets * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    acc_offset = (
        token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d
    )
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    running_acc = tl.load(acc_ptr + acc_offset, mask=dim_mask, other=0.0).to(
        tl.float32
    )

    seq_len = tl.load(seq_lens_ptr + token_idx)
    gather_len = tl.load(gather_lens_ptr + token_idx)
    start_pos = seq_len - gather_len
    fp8_mask = offsets < fp8_dim
    rope_mask = (offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        gather_idx = candidate_offset + candidate_idx
        is_valid = gather_idx < gather_len

        if is_valid:
            pos = start_pos + gather_idx
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            physical_block = tl.load(
                block_table_ptr + token_idx * stride_block_table_t + block_in_seq
            )
            cache_block_ptr = (
                k_cache_ptr + physical_block.to(tl.int64) * block_stride
            )
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv, axis=0) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    tl.store(max_score_ptr + state_offset, running_max)
    tl.store(denom_ptr + state_offset, running_denom)
    tl.store(acc_ptr + acc_offset, running_acc, mask=dim_mask)


def accumulate_fp8ds_paged_sparse_mla_attention_chunk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    scale: float,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    candidate_offset: int,
    num_candidates: int,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert q.shape[-1] == 512
    assert seq_lens.shape[0] == q.shape[0]
    assert gather_lens.shape[0] == q.shape[0]
    assert block_table.shape[0] == q.shape[0]
    assert max_score.shape == q.shape[:2]
    assert denom.shape == q.shape[:2]
    assert acc.shape == q.shape
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert k_cache.dtype == torch.uint8
    assert q.is_cuda and k_cache.is_cuda
    assert seq_lens.is_cuda and gather_lens.is_cuda and block_table.is_cuda
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, num_heads, head_dim = q.shape
    block_d = min(1024, triton.next_power_of_2(head_dim))
    grid = (num_tokens, num_heads)
    _accumulate_fp8ds_paged_attention_chunk_kernel[grid](
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        candidate_offset,
        scale,
        BLOCK_D=block_d,
        num_warps=8,
    )


@triton.jit
def _finish_attention_state_kernel(
    max_score_ptr,
    denom_ptr,
    acc_ptr,
    output_ptr,
    lse_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_head = tl.program_id(0)
    block_d = tl.program_id(1)
    token_idx = token_head // num_heads
    head_idx = token_head - token_idx * num_heads
    offsets = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim

    state_offset = token_idx * stride_state_t + head_idx * stride_state_h
    running_max = tl.load(max_score_ptr + state_offset)
    running_denom = tl.load(denom_ptr + state_offset)
    is_valid = running_denom > 0.0
    inv_denom = tl.where(is_valid, 1.0 / running_denom, 0.0)
    subset_lse = tl.where(
        is_valid,
        running_max + tl.log(running_denom),
        -float("inf"),
    )

    acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_idx * stride_acc_h
        + offsets * stride_acc_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    subset_output = acc * inv_denom
    tl.store(
        output_ptr
        + token_idx * stride_output_t
        + head_idx * stride_output_h
        + offsets * stride_output_d,
        subset_output,
        mask=dim_mask,
    )
    if block_d == 0:
        tl.store(
            lse_ptr + token_idx * stride_lse_t + head_idx * stride_lse_h,
            subset_lse,
        )


def finish_gathered_sparse_mla_attention(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    assert max_score.shape == denom.shape
    assert acc.shape[:2] == max_score.shape
    assert output.shape == acc.shape
    assert lse.shape == max_score.shape
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert output.dtype == torch.float32
    assert lse.dtype == torch.float32
    assert max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert output.is_cuda and lse.is_cuda

    num_tokens, num_heads, head_dim = acc.shape
    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens * num_heads, triton.cdiv(head_dim, block_d))
    _finish_attention_state_kernel[grid](
        max_score,
        denom,
        acc,
        output,
        lse,
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_heads,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
