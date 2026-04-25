# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for portable sparse MLA Triton kernels."""

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk,
    accumulate_gathered_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk,
    finish_gathered_sparse_mla_attention,
    merge_sparse_mla_subset_with_sink,
    merge_two_sparse_mla_subsets_with_sink,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


def _scores(
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
    scores = _scores(q, kv, valid_tokens, scale)
    lse = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - lse[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)
    output = torch.einsum("bht,btd->bhd", weights, kv.float())
    has_valid = valid_tokens.any(dim=-1)
    output = torch.where(
        has_valid[:, None, None],
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
    scores = _scores(q, kv, valid_tokens, scale)
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
    sink_weight = torch.nan_to_num(torch.exp(sink - merge_max))
    numerator = torch.einsum("bht,btd->bhd", weights, kv.float())
    denom = weights.sum(dim=-1) + sink_weight
    return numerator / denom[:, :, None]


def _golden_merge_with_sink(
    subset_outputs: list[torch.Tensor],
    subset_lses: list[torch.Tensor],
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    assert len(subset_outputs) == len(subset_lses)
    sink = attn_sink[None, :].float()
    max_lse = sink
    for subset_lse in subset_lses:
        max_lse = torch.maximum(max_lse, subset_lse.float())

    sink_weight = torch.nan_to_num(torch.exp(sink - max_lse))
    denom = sink_weight.clone()
    numerator = torch.zeros_like(subset_outputs[0], dtype=torch.float32)
    for subset_output, subset_lse in zip(subset_outputs, subset_lses):
        weight = torch.nan_to_num(torch.exp(subset_lse.float() - max_lse))
        denom = denom + weight
        numerator = numerator + subset_output.float() * weight[:, :, None]
    return numerator / denom[:, :, None]


def _finish_state(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(acc)
    lse = torch.empty_like(max_score)
    finish_gathered_sparse_mla_attention(max_score, denom, acc, output, lse)
    return output, lse


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
    flat_block[token_data_start:token_data_start + _FP8_DIM] = fp8_values.view(
        torch.uint8
    )
    flat_block[
        token_data_start + _FP8_DIM:token_data_start + _TOKEN_DATA_SIZE
    ] = rope.view(torch.uint8)

    encoded_scales = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat_block[token_scale_start:token_scale_start + encoded_scales.numel()] = (
        encoded_scales
    )
    flat_block[
        token_scale_start + encoded_scales.numel():token_scale_start + _SCALE_DIM
    ] = 127
    return torch.cat([expected_nope, rope.float()]).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_single_subset_lse_merge_with_sink_matches_reference() -> None:
    torch.manual_seed(0)
    subset_output = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    subset_lse = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    subset_lse[1, 2] = float("-inf")
    sink = torch.tensor([-0.5, 0.25, 1.0, -1.5], device="cuda")
    output = torch.empty(3, 4, 9, device="cuda", dtype=torch.bfloat16)

    merge_sparse_mla_subset_with_sink(subset_output, subset_lse, sink, output)
    expected = _golden_merge_with_sink([subset_output], [subset_lse], sink)

    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_two_subset_lse_merge_with_sink_matches_reference() -> None:
    torch.manual_seed(1)
    out0 = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    out1 = torch.randn(3, 4, 9, device="cuda", dtype=torch.float32)
    lse0 = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    lse1 = torch.randn(3, 4, device="cuda", dtype=torch.float32)
    lse0[1, 2] = float("-inf")
    lse1[2, 1] = float("-inf")
    sink = torch.tensor([-0.5, 0.25, 1.0, -1.5], device="cuda")
    output = torch.empty(3, 4, 9, device="cuda", dtype=torch.bfloat16)

    merge_two_sparse_mla_subsets_with_sink(out0, lse0, out1, lse1, sink, output)
    expected = _golden_merge_with_sink([out0, out1], [lse0, lse1], sink)

    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("head_dim", [16, 512])
def test_gathered_bf16_attention_chunks_match_reference(head_dim: int) -> None:
    torch.manual_seed(2)
    scale = 0.125
    q = torch.randn(2, 1, 3, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 5, head_dim, device="cuda", dtype=torch.bfloat16)
    slot_ids = torch.tensor(
        [[0, 1, -1, 3, 4], [5, -1, 7, 8, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    max_score = torch.full((2, 3), float("-inf"), device="cuda")
    denom = torch.zeros((2, 3), device="cuda")
    acc = torch.zeros((2, 3, head_dim), device="cuda")

    accumulate_gathered_sparse_mla_attention_chunk(
        q, kv[:, :2], lens, scale, max_score, denom, acc,
        candidate_offset=0, slot_ids=slot_ids[:, :2]
    )
    accumulate_gathered_sparse_mla_attention_chunk(
        q, kv[:, 2:], lens, scale, max_score, denom, acc,
        candidate_offset=2, slot_ids=slot_ids[:, 2:]
    )
    output, lse = _finish_state(max_score, denom, acc)

    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid_tokens = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = _golden_no_sink_attention(
        q, kv, valid_tokens, scale
    )
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_indexed_bf16_prefill_chunks_match_sink_reference() -> None:
    torch.manual_seed(3)
    q = torch.randn(5, 3, 16, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16)
    kv_flat = kv.reshape(-1, q.shape[-1])
    indices = torch.tensor(
        [
            [0, 3, -1, 5, 3, 1],
            [4, -1, 2, 2, 1, 8],
            [-1, -1, -1, -1, -1, -1],
            [8, 0, 9, -1, 7, 4],
            [13, 12, 0, 12, -1, 3],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    lens = torch.tensor([5, 4, 0, 6, 5], dtype=torch.int32, device="cuda")
    sink = torch.tensor([-0.5, 1.0, 0.25], dtype=torch.float32, device="cuda")
    scale = 0.375
    output = torch.empty_like(q)

    for token_start in (0, 2, 4):
        token_end = min(token_start + 2, q.shape[0])
        q_chunk = q[token_start:token_end]
        max_score = torch.full(
            (q_chunk.shape[0], q.shape[1]), float("-inf"), device="cuda"
        )
        denom = torch.zeros_like(max_score)
        acc = torch.zeros_like(q_chunk, dtype=torch.float32)
        for index_start in (0, 3):
            index_end = min(index_start + 3, indices.shape[-1])
            accumulate_indexed_sparse_mla_attention_chunk(
                q_chunk,
                kv_flat,
                indices[token_start:token_end, index_start:index_end],
                lens[token_start:token_end],
                scale,
                max_score,
                denom,
                acc,
                candidate_offset=index_start,
            )
        subset_output, subset_lse = _finish_state(max_score, denom, acc)
        merge_sparse_mla_subset_with_sink(
            subset_output, subset_lse, sink, output[token_start:token_end]
        )

    offsets = torch.arange(indices.shape[-1], device="cuda")
    valid = (offsets[None, :] < lens[:, None]) & (indices >= 0)
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices)).long()
    gathered = kv_flat[safe_indices]
    expected = _golden_sink_attention(q, gathered, valid, scale, sink)
    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fp8ds_global_slot_attention_chunks_match_reference() -> None:
    torch.manual_seed(4)
    block_size = 4
    k_cache = torch.zeros(
        3,
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
        [[0, 3, -1, 8, 1], [7, -1, 4, 0, 8]],
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
        q, k_cache, slot_ids[:, :2], lens, block_size, scale,
        max_score, denom, acc, candidate_offset=0
    )
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk(
        q, k_cache, slot_ids[:, 2:], lens, block_size, scale,
        max_score, denom, acc, candidate_offset=2
    )
    output, lse = _finish_state(max_score, denom, acc)

    gathered = torch.zeros(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    for token_idx in range(slot_ids.shape[0]):
        for topk_idx in range(slot_ids.shape[1]):
            slot = int(slot_ids[token_idx, topk_idx].item())
            if slot >= 0:
                gathered[token_idx, topk_idx] = expected_by_slot[slot]
    offsets = torch.arange(slot_ids.shape[1], device="cuda")
    valid = (offsets[None, :] < lens[:, None]) & (slot_ids >= 0)
    expected_output, expected_lse = _golden_no_sink_attention(q, gathered, valid, scale)
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fp8ds_paged_attention_with_sink_matches_reference() -> None:
    torch.manual_seed(5)
    block_size = 4
    k_cache = torch.zeros(
        3,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor([[1, 0, 2]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([7], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    q = torch.randn(1, 1, 3, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.tensor([-0.25, 0.5, 1.25], device="cuda")
    scale = 0.0625

    gathered = torch.zeros(1, 4, 512, device="cuda", dtype=torch.bfloat16)
    expected_by_slot: dict[int, torch.Tensor] = {}
    start_pos = int(seq_lens[0].item() - gather_lens[0].item())
    for gather_idx in range(int(gather_lens[0].item())):
        pos = start_pos + gather_idx
        physical_block = int(block_table[0, pos // block_size].item())
        slot = physical_block * block_size + pos % block_size
        expected_by_slot.setdefault(
            slot, _write_fp8_ds_mla_token(k_cache, slot, block_size)
        )
        gathered[0, gather_idx] = expected_by_slot[slot]

    max_score = torch.full((1, 3), float("-inf"), device="cuda")
    denom = torch.zeros((1, 3), device="cuda")
    acc = torch.zeros((1, 3, 512), device="cuda")
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        scale,
        max_score,
        denom,
        acc,
        candidate_offset=0,
        num_candidates=2,
    )
    accumulate_fp8ds_paged_sparse_mla_attention_chunk(
        q,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        scale,
        max_score,
        denom,
        acc,
        candidate_offset=2,
        num_candidates=2,
    )
    subset_output, subset_lse = _finish_state(max_score, denom, acc)
    output = torch.empty(1, 3, 512, device="cuda", dtype=torch.bfloat16)
    merge_sparse_mla_subset_with_sink(subset_output, subset_lse, sink, output)

    valid = torch.ones(1, 4, device="cuda", dtype=torch.bool)
    expected = _golden_sink_attention(q, gathered, valid, scale, sink)
    torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)
