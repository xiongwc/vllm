# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_sparse_mla_attention_dump_enabled,
    is_sparse_mla_reference_attention_enabled,
    sparse_mla_attention_dump_path,
    sparse_mla_reference_query_chunk_size,
    sparse_mla_reference_topk_chunk_size,
)

_SPARSE_MLA_ENV_NAMES = (
    "VLLM_TRITON_MLA_SPARSE",
    "VLLM_TRITON_MLA_SPARSE_DUMP",
    "VLLM_TRITON_MLA_SPARSE_DUMP_PATH",
    "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE",
    "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE",
    "VLLM_SM120_DUMP_DEEPSEEK_V4_ATTENTION",
    "VLLM_SM120_ATTENTION_DUMP_PATH",
    "VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION",
    "VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE",
    "VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE",
)


@contextmanager
def _patched_sparse_mla_env(**updates: str) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _SPARSE_MLA_ENV_NAMES}
    try:
        for name in _SPARSE_MLA_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(updates)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_sparse_mla_reference_env_prefers_new_name() -> None:
    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE="0",
        VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION="1",
    ):
        assert not is_sparse_mla_reference_attention_enabled(torch.device("cpu"))

    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE="1",
        VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION="0",
    ):
        assert is_sparse_mla_reference_attention_enabled(torch.device("cpu"))


def test_sparse_mla_dump_env_prefers_new_name() -> None:
    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE_DUMP="0",
        VLLM_SM120_DUMP_DEEPSEEK_V4_ATTENTION="1",
    ):
        assert not is_sparse_mla_attention_dump_enabled()

    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE_DUMP="1",
        VLLM_SM120_DUMP_DEEPSEEK_V4_ATTENTION="0",
    ):
        assert is_sparse_mla_attention_dump_enabled()


def test_sparse_mla_chunk_env_uses_legacy_aliases() -> None:
    with _patched_sparse_mla_env(
        VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE="17",
        VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE="9",
    ):
        assert sparse_mla_reference_topk_chunk_size() == 17
        assert sparse_mla_reference_query_chunk_size() == 9


def test_sparse_mla_chunk_env_defaults_invalid_values() -> None:
    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="invalid",
        VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="-7",
    ):
        assert sparse_mla_reference_topk_chunk_size() == 512
        assert sparse_mla_reference_query_chunk_size() == 1


def test_sparse_mla_dump_path_prefers_new_name() -> None:
    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE_DUMP_PATH="/tmp/new.jsonl",
        VLLM_SM120_ATTENTION_DUMP_PATH="/tmp/legacy.jsonl",
    ):
        assert sparse_mla_attention_dump_path() == "/tmp/new.jsonl"
