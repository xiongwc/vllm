# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling profiler for DeepSeek V4 SM12x tuning.

This module is intentionally inert unless ``VLLM_DEEPSEEK_V4_PROFILE=1``.
It uses CUDA events around coarse model regions so we can compare decode
component costs without requiring an external Nsight capture for every run.
"""

import os
import threading
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_tls = threading.local()
_step_lock = threading.Lock()
_forward_step = 0


def _profile_enabled() -> bool:
    return bool(envs.VLLM_DEEPSEEK_V4_PROFILE and torch.cuda.is_available())


def _positive_int(value: int | None, default: int) -> int:
    if value is None:
        return default
    return max(1, int(value))


@dataclass
class _TimedRegion:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class _DeepSeekV4ProfileSample:
    def __init__(self, step: int, num_tokens: int):
        self.step = step
        self.num_tokens = num_tokens
        self.regions: list[_TimedRegion] = []
        self.stats: dict[str, str] = {}

    def region(self, name: str):
        return _ProfileRegion(self, name)

    def add_region(
        self,
        name: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        self.regions.append(_TimedRegion(name, start, end))

    def add_stat(self, name: str, value: Any) -> None:
        self.stats[name] = str(value)

    def log(self) -> None:
        if not self.regions:
            return

        self.regions[-1].end.synchronize()

        totals: defaultdict[str, float] = defaultdict(float)
        counts: defaultdict[str, int] = defaultdict(int)
        for region in self.regions:
            elapsed_ms = region.start.elapsed_time(region.end)
            totals[region.name] += elapsed_ms
            counts[region.name] += 1

        model_total = totals.get("model.forward", 0.0)
        entries = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        parts = []
        for name, total_ms in entries:
            count = counts[name]
            avg_ms = total_ms / count
            pct = (total_ms / model_total * 100.0) if model_total > 0 else 0.0
            parts.append(
                f"{name}={total_ms:.3f}ms/{count} "
                f"avg={avg_ms:.3f}ms {pct:.1f}%"
            )

        stats = (
            " stats: "
            + ", ".join(f"{name}={value}" for name, value in sorted(self.stats.items()))
            if self.stats
            else ""
        )
        logger.info(
            "DeepSeek V4 profile %s step=%d tokens=%d:%s %s",
            _worker_label(),
            self.step,
            self.num_tokens,
            stats,
            "; ".join(parts),
        )


class _ProfileRegion:
    def __init__(self, sample: _DeepSeekV4ProfileSample, name: str):
        self.sample = sample
        self.name = name
        self.start: torch.cuda.Event | None = None
        self.end: torch.cuda.Event | None = None

    def __enter__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, *args: Any) -> None:
        assert self.start is not None
        assert self.end is not None
        self.end.record()
        self.sample.add_region(self.name, self.start, self.end)


class _ProfileForward:
    def __init__(self, sample: _DeepSeekV4ProfileSample):
        self.sample = sample
        self.previous_sample: _DeepSeekV4ProfileSample | None = None

    def __enter__(self) -> _DeepSeekV4ProfileSample:
        self.previous_sample = getattr(_tls, "sample", None)
        _tls.sample = self.sample
        self.region = self.sample.region("model.forward")
        self.region.__enter__()
        return self.sample

    def __exit__(self, *args: Any) -> None:
        self.region.__exit__(*args)
        _tls.sample = self.previous_sample
        self.sample.log()


def maybe_profile_deepseek_v4_forward(num_tokens: int):
    if not _profile_enabled():
        return nullcontext()

    global _forward_step
    with _step_lock:
        _forward_step += 1
        step = _forward_step

    warmup = _positive_int(envs.VLLM_DEEPSEEK_V4_PROFILE_WARMUP, 16)
    interval = _positive_int(envs.VLLM_DEEPSEEK_V4_PROFILE_INTERVAL, 128)
    if step <= warmup or step % interval != 0:
        return nullcontext()

    return _ProfileForward(_DeepSeekV4ProfileSample(step, num_tokens))


def deepseek_v4_profile_region(name: str):
    sample = getattr(_tls, "sample", None)
    if sample is None:
        return nullcontext()
    return sample.region(name)


def deepseek_v4_profile_active() -> bool:
    return getattr(_tls, "sample", None) is not None


def deepseek_v4_profile_stat(name: str, value: Any) -> None:
    sample = getattr(_tls, "sample", None)
    if sample is None:
        return
    sample.add_stat(name, value)


def deepseek_v4_profile_lens(name: str, lens: torch.Tensor) -> None:
    sample = getattr(_tls, "sample", None)
    if sample is None:
        return
    if lens.numel() == 0:
        sample.add_stat(name, "empty")
        return

    lens_fp32 = lens.detach().to(torch.float32)
    min_len = int(lens.min().item())
    max_len = int(lens.max().item())
    avg_len = float(lens_fp32.mean().item())
    sample.add_stat(name, f"min={min_len},avg={avg_len:.1f},max={max_len}")


def _worker_label() -> str:
    parts = [f"pid={os.getpid()}"]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            parts.append(f"rank={torch.distributed.get_rank()}")
            parts.append(f"world={torch.distributed.get_world_size()}")
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        parts.append(f"cuda={device}")
        parts.append(f"gpu={torch.cuda.get_device_name(device)}")
    return " ".join(parts)
