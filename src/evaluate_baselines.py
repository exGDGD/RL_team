from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.baselines import (
    EASLikePolicy,
    RandomPolicy,
    RoundRobinPolicy,
    SJFLikePolicy,
    run_episode,
)
from src.env import CoreType, SchedulerEnv, WorkloadScenario


@dataclass(frozen=True)
class EvalConfig:
    seeds: tuple[int, ...] = tuple(range(5))
    arrival_rate: float = 0.6
    episode_time: float = 160.0
    max_tasks: int = 96
    queue_size: int = 8


def main() -> None:
    config = EvalConfig()
    scenarios = [
        WorkloadScenario.BALANCED,
        WorkloadScenario.UI_HEAVY,
        WorkloadScenario.BG_HEAVY,
        WorkloadScenario.BURST_STRESS,
    ]
    policies = [
        RandomPolicy(seed=0),
        RoundRobinPolicy(),
        SJFLikePolicy(),
        EASLikePolicy(),
    ]
    rows = []
    for scenario in scenarios:
        for policy in policies:
            results = [
                run_episode(_make_env(scenario, seed, config), policy, seed=seed)
                for seed in config.seeds
            ]
            rows.append(_summarize_results(scenario.value, policy.name, results))

    _print_table(rows)


def _make_env(scenario: WorkloadScenario, seed: int, config: EvalConfig) -> SchedulerEnv:
    return SchedulerEnv(
        core_config={CoreType.P: 2, CoreType.E: 2},
        queue_size=config.queue_size,
        workload_scenario=scenario,
        arrival_rate=config.arrival_rate,
        episode_time=config.episode_time,
        max_tasks=config.max_tasks,
        seed=seed,
    )


def _summarize_results(scenario: str, policy: str, results: list) -> dict[str, object]:
    return {
        "scenario": scenario,
        "policy": policy,
        "reward": _mean_std(result.total_reward for result in results),
        "completed": _mean_std(result.metrics.completed_tasks for result in results),
        "throughput": _mean_std(result.metrics.throughput for result in results),
        "energy": _mean_std(result.metrics.total_energy for result in results),
        "mean_response": _mean_std(
            result.metrics.mean_response_time for result in results
        ),
        "mean_turnaround": _mean_std(
            result.metrics.mean_turnaround_time for result in results
        ),
        "starvation": _mean_std(result.metrics.starvation_rate for result in results),
        "util": _mean_std(result.metrics.mean_utilization for result in results),
    }


def _mean_std(values: Iterable[float | int | None]) -> str:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return "-"
    mean = float(np.mean(numeric_values))
    std = float(np.std(numeric_values))
    return f"{mean:.3f}+/-{std:.3f}"


def _print_table(rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    columns = [
        "scenario",
        "policy",
        "reward",
        "completed",
        "throughput",
        "energy",
        "mean_response",
        "mean_turnaround",
        "starvation",
        "util",
    ]
    widths = {
        column: max(len(column), *(len(_format_value(row[column])) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_format_value(row[column]).ljust(widths[column]) for column in columns))


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


if __name__ == "__main__":
    main()
