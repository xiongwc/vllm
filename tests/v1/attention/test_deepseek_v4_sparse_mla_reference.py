# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the DeepSeek V4 sparse MLA reference path."""

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_reference import (
    accumulate_reference_attention_chunk,
    finish_reference_attention_no_sink,
    merge_reference_attention_with_sink,
    new_reference_attention_state,
    reference_attention_no_sink,
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
