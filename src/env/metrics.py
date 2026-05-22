from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable, Mapping

import numpy as np

from .core import Core
from .task import Task


@dataclass(frozen=True)
class EpisodeMetrics:
    total_tasks: int
    completed_tasks: int
    makespan: float
    throughput: float
    total_energy: float
    mean_response_time: float | None
    p95_response_time: float | None
    p99_response_time: float | None
    mean_turnaround_time: float | None
    p95_turnaround_time: float | None
    p99_turnaround_time: float | None
    mean_ready_wait_time: float | None
    p95_ready_wait_time: float | None
    starvation_rate: float | None
    mean_utilization: float
    per_core_utilization: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_episode_metrics(
    *,
    tasks: Iterable[Task],
    cores: Mapping[str, Core],
    now: float,
    starvation_threshold: float = 100.0,
) -> EpisodeMetrics:
    task_list = list(tasks)
    completed = [task for task in task_list if task.done]
    response_times = [
        task.response_time() for task in task_list if task.response_time() is not None
    ]
    turnaround_times = [
        task.turnaround_time() for task in completed if task.turnaround_time() is not None
    ]
    ready_wait_times = [task.total_ready_wait_time for task in completed]
    max_ready_wait_times = [task.max_ready_wait_time for task in completed]
    per_core_utilization = {
        core_id: _safe_divide(core.busy_time, now) for core_id, core in cores.items()
    }

    return EpisodeMetrics(
        total_tasks=len(task_list),
        completed_tasks=len(completed),
        makespan=float(now),
        throughput=_safe_divide(len(completed), now),
        total_energy=float(sum(core.accumulated_energy for core in cores.values())),
        mean_response_time=_mean(response_times),
        p95_response_time=_percentile(response_times, 95),
        p99_response_time=_percentile(response_times, 99),
        mean_turnaround_time=_mean(turnaround_times),
        p95_turnaround_time=_percentile(turnaround_times, 95),
        p99_turnaround_time=_percentile(turnaround_times, 99),
        mean_ready_wait_time=_mean(ready_wait_times),
        p95_ready_wait_time=_percentile(ready_wait_times, 95),
        starvation_rate=_starvation_rate(max_ready_wait_times, starvation_threshold),
        mean_utilization=_mean(per_core_utilization.values()) or 0.0,
        per_core_utilization=per_core_utilization,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.percentile(values, percentile))


def _starvation_rate(values: Iterable[float], threshold: float) -> float | None:
    values = list(values)
    if not values:
        return None
    starved = sum(wait_time > threshold for wait_time in values)
    return float(starved / len(values))
