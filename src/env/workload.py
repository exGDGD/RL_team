from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .task import LatencyClass, Task


class WorkloadScenario(str, Enum):
    BALANCED = "balanced"
    UI_HEAVY = "ui_heavy"
    BG_HEAVY = "bg_heavy"
    BURST_STRESS = "burst_stress"


@dataclass(frozen=True)
class WorkloadConfig:
    scenario: WorkloadScenario = WorkloadScenario.BALANCED
    arrival_rate: float = 0.15
    max_tasks: int = 64
    episode_time: float = 500.0


class WorkloadGenerator:
    def __init__(self, config: WorkloadConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[Task]:
        tasks: list[Task] = []
        now = 0.0
        for pid in range(self.config.max_tasks):
            now += float(self.rng.exponential(1.0 / self.config.arrival_rate))
            if now > self.config.episode_time:
                break
            tasks.append(self._make_task(pid=pid, arrival_time=now))
        return tasks

    def _make_task(self, pid: int, arrival_time: float) -> Task:
        scenario = self.config.scenario
        if scenario == WorkloadScenario.UI_HEAVY:
            latency_probs = [0.15, 0.35, 0.50]
            cpu_alpha, cpu_beta = 2.0, 5.0
        elif scenario == WorkloadScenario.BG_HEAVY:
            latency_probs = [0.75, 0.20, 0.05]
            cpu_alpha, cpu_beta = 2.5, 2.5
        elif scenario == WorkloadScenario.BURST_STRESS:
            latency_probs = [0.35, 0.40, 0.25]
            cpu_alpha, cpu_beta = 5.0, 1.8
        else:
            latency_probs = [0.45, 0.35, 0.20]
            cpu_alpha, cpu_beta = 2.0, 2.0

        cpu_intensity = float(self.rng.beta(cpu_alpha, cpu_beta))
        latency_class = LatencyClass(
            int(self.rng.choice([0, 1, 2], p=np.array(latency_probs)))
        )
        phase_count = int(self.rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
        burst_scale = 16.0 if scenario == WorkloadScenario.BURST_STRESS else 8.0
        cpu_bursts = [
            float(max(1.0, self.rng.gamma(shape=2.0, scale=burst_scale) * cpu_intensity))
            for _ in range(phase_count)
        ]
        io_waits = [
            float(max(0.5, self.rng.gamma(shape=1.5, scale=6.0) * (1.0 - cpu_intensity)))
            for _ in range(phase_count - 1)
        ]
        return Task(
            pid=pid,
            arrival_time=arrival_time,
            cpu_intensity=cpu_intensity,
            latency_class=latency_class,
            cpu_bursts=cpu_bursts,
            io_waits=io_waits,
        )
