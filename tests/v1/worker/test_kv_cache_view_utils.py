# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.kv_cache_view_utils import view_kv_cache_with_layout


def test_padded_kv_cache_view_uses_block_axis_for_standard_layout():
    num_blocks = 3
    semantic_shape = (2, num_blocks, 2, 1, 2)
    stride_order = (0, 1, 2, 3, 4)
    block_axis = 1
    page_elements = 12
    raw = torch.arange(num_blocks * page_elements, dtype=torch.int16)

    kv_cache = view_kv_cache_with_layout(
        raw_tensor=raw,
        kv_cache_shape=semantic_shape,
        kv_cache_stride_order=stride_order,
        block_axis=block_axis,
        dtype=torch.int16,
        page_size_bytes=page_elements * raw.element_size(),
        page_size_padded=page_elements * raw.element_size(),
    )

    for kv in range(semantic_shape[0]):
        for block in range(num_blocks):
            for token in range(semantic_shape[2]):
                for dim in range(semantic_shape[4]):
                    expected_offset = block * page_elements + kv * 4 + token * 2 + dim
                    assert kv_cache[kv, block, token, 0, dim] == raw[expected_offset]


def test_padded_kv_cache_view_handles_block_first_layout():
    num_blocks = 3
    semantic_shape = (num_blocks, 2, 4)
    stride_order = (0, 1, 2)
    block_axis = 0
    page_elements = 12
    raw = torch.arange(num_blocks * page_elements, dtype=torch.int16)

    kv_cache = view_kv_cache_with_layout(
        raw_tensor=raw,
        kv_cache_shape=semantic_shape,
        kv_cache_stride_order=stride_order,
        block_axis=block_axis,
        dtype=torch.int16,
        page_size_bytes=page_elements * raw.element_size(),
        page_size_padded=page_elements * raw.element_size(),
    )

    for block in range(num_blocks):
        for token in range(semantic_shape[1]):
            for dim in range(semantic_shape[2]):
                expected_offset = block * page_elements + token * 4 + dim
                assert kv_cache[block, token, dim] == raw[expected_offset]
