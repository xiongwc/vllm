# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend


def get_kv_cache_stride_order(
    attn_backend: type[AttentionBackend],
    kv_cache_shape: Sequence[int],
) -> tuple[int, ...]:
    try:
        stride_order = attn_backend.get_kv_cache_stride_order()
        assert len(stride_order) == len(kv_cache_shape)
        return stride_order
    except (AttributeError, NotImplementedError):
        return tuple(range(len(kv_cache_shape)))


def get_kv_cache_block_axis(
    attn_backend: type[AttentionBackend],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str,
) -> int:
    try:
        return attn_backend.get_kv_cache_block_axis(
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
    except (AttributeError, NotImplementedError):
        return 0


def _paged_kv_cache_strides(
    ordered_shape: Sequence[int],
    ordered_block_axis: int,
    page_stride: int,
) -> tuple[int, ...]:
    """Build strides for a block-paged raw allocation.

    The block axis crosses page boundaries. All other physical axes are laid out
    contiguously inside a page, regardless of whether they appear before or
    after the block axis in the backend's stride order.
    """
    strides = [0] * len(ordered_shape)
    inner_stride = 1
    for axis in reversed(range(len(ordered_shape))):
        if axis == ordered_block_axis:
            continue
        strides[axis] = inner_stride
        inner_stride *= ordered_shape[axis]

    assert page_stride >= inner_stride
    strides[ordered_block_axis] = page_stride
    return tuple(strides)


def view_kv_cache_with_layout(
    *,
    raw_tensor: torch.Tensor,
    kv_cache_shape: Sequence[int],
    kv_cache_stride_order: Sequence[int],
    block_axis: int,
    dtype: torch.dtype,
    page_size_bytes: int,
    page_size_padded: int | None,
) -> torch.Tensor:
    """View a raw KV allocation as the backend's semantic cache shape."""
    ordered_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = tuple(
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    )

    raw_tensor = raw_tensor.view(dtype)
    if page_size_padded is None:
        ordered_kv_cache = raw_tensor.view(ordered_shape)
    else:
        ordered_block_axis = kv_cache_stride_order.index(block_axis)
        dtype_size = get_dtype_size(dtype)
        page_stride = page_size_bytes // dtype_size
        strides = _paged_kv_cache_strides(
            ordered_shape,
            ordered_block_axis,
            page_stride,
        )
        ordered_kv_cache = torch.as_strided(
            raw_tensor,
            size=ordered_shape,
            stride=strides,
        )

    return ordered_kv_cache.permute(*inv_order)
