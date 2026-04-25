# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the DeepSeek V4 sparse MLA reference path."""

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk,
    accumulate_gathered_sparse_mla_attention_chunk,
    finish_gathered_sparse_mla_attention,
    merge_two_sparse_mla_subsets_with_sink,
)
from vllm.v1.attention.backends.mla.sparse_mla_reference import (
    accumulate_reference_attention_chunk,
    finish_reference_attention_no_sink,
    merge_reference_attention_with_sink,
    new_reference_attention_state,
    reference_attention_no_sink,
    reference_sparse_mla_prefill,
    sink_aware_reference_attention,
)
from vllm.v1.attention.ops.deepseek_v4_ops import dequantize_global_slots_k_cache


_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


def _masked_scores(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_bhd = q[:, 0].float() if q.dim() == 4 else q.float()
    scores = torch.einsum("bhd,btd->bht", q_bhd, kv.float()) * scale
    return scores.masked_fill(~valid_tokens[:, None, :], float("-inf"))


def _golden_no_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _masked_scores(q, kv, valid_tokens, scale)
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)
    output = torch.einsum("bht,btd->bhd", weights, kv.float())
    valid = valid_tokens.any(dim=-1)
    output = torch.where(
        valid[:, None, None],
        output,
        torch.zeros((), dtype=output.dtype, device=output.device),
    )
    return output, lse


def _golden_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    scores = _masked_scores(q, kv, valid_tokens, scale)
    sink = attn_sink[None, :].float()
    score_max = scores.amax(dim=-1)
    merge_max = torch.maximum(score_max, sink)

    weights = torch.exp(scores - merge_max[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)

    sink_weight = torch.exp(sink - merge_max)
    sink_weight = torch.nan_to_num(sink_weight)
    denom = weights.sum(dim=-1) + sink_weight
    numerator = torch.einsum("bht,btd->bhd", weights, kv.float())
    return numerator / denom[:, :, None]


def _chunked_no_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_bhd, max_score, denom, acc = new_reference_attention_state(q)
    for chunk_start in range(0, kv.shape[1], chunk_size):
        chunk_end = min(chunk_start + chunk_size, kv.shape[1])
        max_score, denom, acc = accumulate_reference_attention_chunk(
            q_bhd=q_bhd,
            kv=kv[:, chunk_start:chunk_end],
            valid_tokens=valid_tokens[:, chunk_start:chunk_end],
            max_score=max_score,
            denom=denom,
            acc=acc,
            scale=scale,
        )
    return finish_reference_attention_no_sink(max_score, denom, acc)


def _write_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> torch.Tensor:
    block_idx = slot // block_size
    block_offset = slot % block_size

    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17)
        - 8
    ) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1],
        device=k_cache.device,
        dtype=torch.float32,
    )
    scales = torch.exp2(scale_exponents)
    scale_per_dim = scales.repeat_interleave(64)

    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    expected_nope = fp8_values.float() * scale_per_dim
    rope = (
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device)
        + float(slot) / 16.0
    ).to(torch.bfloat16)

    flat_block = k_cache[block_idx].view(-1)
    token_data_start = block_offset * _TOKEN_DATA_SIZE
    token_scale_start = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat_block[token_data_start : token_data_start + _FP8_DIM] = fp8_values.view(
        torch.uint8
    )
    flat_block[
        token_data_start + _FP8_DIM : token_data_start + _TOKEN_DATA_SIZE
    ] = rope.view(torch.uint8)

    encoded_scales = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat_block[token_scale_start : token_scale_start + encoded_scales.numel()] = (
        encoded_scales
    )
    flat_block[
        token_scale_start + encoded_scales.numel() : token_scale_start + _SCALE_DIM
    ] = 127

    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


def test_reference_attention_no_sink_matches_logsumexp() -> None:
    torch.manual_seed(0)
    scale = 0.25
    q = torch.randn(3, 4, 5)
    kv = torch.randn(3, 6, 5)
    valid_tokens = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, False, False, False, False, False],
            [True, False, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    output, lse = reference_attention_no_sink(q, kv, valid_tokens, scale)
    expected_output, expected_lse = _golden_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)


def test_sink_aware_reference_attention_matches_dense_golden() -> None:
    torch.manual_seed(1)
    scale = 0.125
    q = torch.randn(3, 1, 4, 5)
    kv = torch.randn(3, 6, 5)
    valid_tokens = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, False, False, False, False, False],
            [False, True, True, False, True, True],
        ],
        dtype=torch.bool,
    )
    sink = torch.tensor([-1.0, 0.25, 1.5, -0.5])
    output = torch.empty(3, 4, 5)
    sink_aware_reference_attention(q, kv, valid_tokens, scale, sink, output)
    expected = _golden_sink_attention(q, kv, valid_tokens, scale, sink)

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


def test_lse_merge_with_sink_matches_concatenated_attention() -> None:
    torch.manual_seed(2)
    scale = 0.2
    q = torch.randn(4, 3, 7)
    compressed_kv = torch.randn(4, 5, 7)
    swa_kv = torch.randn(4, 3, 7)
    compressed_kv[:, 1] = compressed_kv[:, 0]
    swa_kv[:, 2] = compressed_kv[:, 0]
    compressed_valid = torch.tensor(
        [
            [True, True, False, True, False],
            [False, False, False, False, False],
            [True, False, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    swa_valid = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    sink = torch.tensor([-0.25, 0.75, 1.25])
    output = torch.empty(4, 3, 7)
    comp_output, comp_lse = reference_attention_no_sink(
        q,
        compressed_kv,
        compressed_valid,
        scale,
    )
    swa_output, swa_lse = reference_attention_no_sink(q, swa_kv, swa_valid, scale)
    merge_reference_attention_with_sink(
        subset_outputs=[comp_output, swa_output],
        subset_lses=[comp_lse, swa_lse],
        attn_sink=sink,
        output=output,
    )

    expected = _golden_sink_attention(
        q,
        torch.cat([compressed_kv, swa_kv], dim=1),
        torch.cat([compressed_valid, swa_valid], dim=1),
        scale,
        sink,
    )
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(output[3], torch.zeros_like(output[3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_lse_merge_with_sink_matches_reference() -> None:
    torch.manual_seed(5)
    comp_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    swa_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    comp_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    swa_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    comp_lse[1, 2] = float("-inf")
    swa_lse[2, 1] = float("-inf")
    sink = torch.tensor([-0.5, 0.25, 1.0, -1.5], device="cuda")

    output = torch.empty(3, 4, 9, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    merge_two_sparse_mla_subsets_with_sink(
        subset0_output=comp_output,
        subset0_lse=comp_lse,
        subset1_output=swa_output,
        subset1_lse=swa_lse,
        attn_sink=sink,
        output=output,
    )
    merge_reference_attention_with_sink(
        subset_outputs=[comp_output, swa_output],
        subset_lses=[comp_lse, swa_lse],
        attn_sink=sink,
        output=expected,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_dim", [16, 512])
def test_triton_gathered_attention_chunk_matches_reference(head_dim: int) -> None:
    torch.manual_seed(6)
    scale = 0.125
    q = torch.randn(2, 1, 3, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    slot_ids = torch.tensor(
        [
            [0, 1, -1, 3, 4],
            [5, -1, 7, 8, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, head_dim), device="cuda")

    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv[:, :2],
        slot_ids=slot_ids[:, :2],
        lens=lens,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv[:, 2:],
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        kv,
        valid_tokens,
        scale,
    )
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_gathered_attention_chunk_matches_reference_without_slot_ids() -> None:
    torch.manual_seed(8)
    scale = 0.2
    q = torch.randn(3, 1, 2, 32, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 6, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([6, 3, 0], dtype=torch.int32, device="cuda")
    max_score = torch.full((3, 2), float("-inf"), device="cuda")
    denom = torch.zeros((3, 2), device="cuda")
    acc = torch.zeros((3, 2, 32), device="cuda")

    accumulate_gathered_sparse_mla_attention_chunk(
        q=q,
        kv=kv,
        slot_ids=None,
        lens=lens,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(kv.shape[1], device="cuda")
    valid_tokens = offsets[None, :] < lens[:, None]
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        kv,
        valid_tokens,
        scale,
    )
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_dequantize_global_slots_k_cache_fp8_ds_mla_layout() -> None:
    block_size = 4
    num_blocks = 2
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 3, 4)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 4],
            [4, 0, 3, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    output = torch.empty(2, 4, 512, dtype=torch.bfloat16, device="cuda")
    dequantize_global_slots_k_cache(output, k_cache, slot_ids, block_size)

    expected = torch.zeros_like(output)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                expected[token_idx, topk_idx] = expected_by_slot[slot]

    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)

    output_from_3d_indices = torch.empty_like(output)
    dequantize_global_slots_k_cache(
        output_from_3d_indices,
        k_cache,
        slot_ids.unsqueeze(1),
        block_size,
    )
    torch.testing.assert_close(
        output_from_3d_indices.float(),
        expected.float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_fp8ds_global_slots_attention_chunk_matches_reference() -> None:
    torch.manual_seed(10)
    block_size = 4
    num_blocks = 3
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_by_slot = {
        slot: _write_fp8_ds_mla_token(k_cache, slot, block_size)
        for slot in (0, 1, 3, 4, 7, 8)
    }
    slot_ids = torch.tensor(
        [
            [0, 3, -1, 8, 1],
            [7, -1, 4, 0, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625

    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, :2],
        lens=lens,
        block_size=block_size,
        candidate_offset=0,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        slot_ids=slot_ids[:, 2:],
        lens=lens,
        block_size=block_size,
        candidate_offset=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                gathered[token_idx, topk_idx] = expected_by_slot[slot]
    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_triton_fp8ds_paged_attention_chunk_matches_reference() -> None:
    torch.manual_seed(12)
    block_size = 4
    k_cache = torch.zeros(
        3,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [
            [1, 0, 2],
            [2, 1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([6, 9], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    scale = 0.0625

    gathered = torch.zeros(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    expected_by_slot: dict[int, torch.Tensor] = {}
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = int(block_table[token_idx, block_idx].item())
            slot = physical_block * block_size + block_offset
            expected_by_slot.setdefault(
                slot,
                _write_fp8_ds_mla_token(k_cache, slot, block_size),
            )
            gathered[token_idx, gather_idx] = expected_by_slot[slot]

    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=2,
        num_candidates=2,
        scale=scale,
        max_score=max_score,
        denom=denom,
        acc=acc,
    )

    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(
        max_score=max_score,
        denom=denom,
        acc=acc,
        output=output,
        lse=lse,
    )

    offsets = torch.arange(gathered.shape[1], device="cuda")
    valid_tokens = offsets[None, :] < gather_lens[:, None]
    expected_output, expected_lse = reference_attention_no_sink(
        q,
        gathered,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("topk_chunk_size", "query_chunk_size"),
    [(1, 1), (2, 3), (5, 2)],
)
def test_reference_sparse_mla_prefill_matches_dense_golden(
    topk_chunk_size: int,
    query_chunk_size: int,
) -> None:
    torch.manual_seed(4)
    scale = 0.375
    q = torch.randn(4, 2, 3)
    kv = torch.randn(2, 5, 3)
    combined_indices = torch.tensor(
        [
            [0, 3, -1, 5, 3],
            [4, -1, 2, 2, 1],
            [-1, -1, -1, -1, -1],
            [8, 0, 9, -1, 7],
        ],
        dtype=torch.int64,
    )
    combined_lens = torch.tensor([4, 3, 0, 5], dtype=torch.int32)
    sink = torch.tensor([-0.5, 1.0])
    output = torch.empty_like(q)

    reference_sparse_mla_prefill(
        q=q,
        kv=kv,
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        scale=scale,
        attn_sink=sink,
        output=output,
        topk_chunk_size=topk_chunk_size,
        query_chunk_size=query_chunk_size,
    )

    kv_flat = kv.reshape(-1, q.shape[-1])
    offsets = torch.arange(combined_indices.shape[-1])
    valid_tokens = (offsets[None, :] < combined_lens[:, None]) & (
        combined_indices >= 0
    )
    safe_indices = torch.where(
        valid_tokens,
        combined_indices,
        torch.zeros((), dtype=combined_indices.dtype),
    ).long()
    gathered_kv = kv_flat[safe_indices]
    expected = _golden_sink_attention(q, gathered_kv, valid_tokens, scale, sink)

    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_chunked_reference_accumulation_matches_one_shot(chunk_size: int) -> None:
    torch.manual_seed(3)
    scale = 0.3
    q = torch.randn(3, 2, 4)
    kv = torch.randn(3, 9, 4)
    valid_tokens = torch.tensor(
        [
            [True, False, True, True, False, False, True, False, True],
            [False, False, False, False, False, False, False, False, False],
            [True, True, True, False, True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    output, lse = _chunked_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
        chunk_size,
    )
    expected_output, expected_lse = _golden_no_sink_attention(
        q,
        kv,
        valid_tokens,
        scale,
    )

    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
