# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable sparse MLA fallback."""

import os

import torch

_TRITON_MLA_SPARSE_ENV = "VLLM_TRITON_MLA_SPARSE"
_TRITON_MLA_SPARSE_DUMP_ENV = "VLLM_TRITON_MLA_SPARSE_DUMP"
_TRITON_MLA_SPARSE_DUMP_PATH_ENV = "VLLM_TRITON_MLA_SPARSE_DUMP_PATH"
_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"

_LEGACY_SM120_ATTENTION_DUMP_ENV = "VLLM_SM120_DUMP_DEEPSEEK_V4_ATTENTION"
_LEGACY_SM120_ATTENTION_DUMP_PATH_ENV = "VLLM_SM120_ATTENTION_DUMP_PATH"
_LEGACY_SM120_REFERENCE_ATTENTION_ENV = (
    "VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION"
)
_LEGACY_SM120_REFERENCE_TOPK_CHUNK_ENV = "VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE"
_LEGACY_SM120_REFERENCE_QUERY_CHUNK_ENV = "VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE"

_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    return None


def _is_sm12x_device(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] == 12


def is_sparse_mla_attention_dump_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_DUMP_ENV)
    if configured is not None:
        return configured
    return _optional_env_flag(_LEGACY_SM120_ATTENTION_DUMP_ENV) or False


def is_sparse_mla_reference_attention_enabled(device: torch.device) -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ENV)
    if configured is not None:
        return configured
    legacy_configured = _optional_env_flag(_LEGACY_SM120_REFERENCE_ATTENTION_ENV)
    if legacy_configured is not None:
        return legacy_configured
    return _is_sm12x_device(device)


def sparse_mla_attention_dump_path() -> str:
    return (
        os.getenv(_TRITON_MLA_SPARSE_DUMP_PATH_ENV)
        or os.getenv(_LEGACY_SM120_ATTENTION_DUMP_PATH_ENV)
        or "/tmp/deepseek_v4_triton_mla_sparse_dump.jsonl"
    )


def sparse_mla_reference_topk_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV) or os.getenv(
        _LEGACY_SM120_REFERENCE_TOPK_CHUNK_ENV
    )
    if raw_value is None:
        return 256
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 256


def sparse_mla_reference_query_chunk_size() -> int:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV) or os.getenv(
        _LEGACY_SM120_REFERENCE_QUERY_CHUNK_ENV
    )
    if raw_value is None:
        return 256
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 256
