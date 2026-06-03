from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class LatencyClass(IntEnum):
    BEST_EFFORT = 0
    SOFT_RT = 1
    HARD_RT = 2


@dataclass
class Task:
    pid: int
    arrival_time: float
    cpu_intensity: float
    latency_class: LatencyClass
    cpu_bursts: list[float]
    io_waits: list[float] = field(default_factory=list)
    cpu_progress: float = 0.0
    current_burst_idx: int = 0
    current_burst_elapsed: float = 0.0
    preemptions: int = 0
    ready_since: float | None = None
    first_started_at: float | None = None
    completed_at: float | None = None
    accumulated_energy_cost: float = 0.0
    accumulated_starvation_cost: float = 0.0
    total_ready_wait_time: float = 0.0
    max_ready_wait_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.cpu_bursts:
            raise ValueError("Task must have at least one CPU burst.")
        if len(self.io_waits) not in {0, len(self.cpu_bursts) - 1}:
            raise ValueError("io_waits must be empty or one shorter than cpu_bursts.")
        if not 0.0 <= self.cpu_intensity <= 1.0:
            raise ValueError("cpu_intensity must be in [0, 1].")

    @property
    def done(self) -> bool:
        return self.completed_at is not None

    @property
    def total_cpu_required(self) -> float:
        return sum(self.cpu_bursts)

    @property
    def current_cpu_burst(self) -> float:
        """Remaining CPU time in the current burst (full burst minus any work
        already consumed by a preempted run)."""
        return self.cpu_bursts[self.current_burst_idx] - self.current_burst_elapsed

    @property
    def remaining_cpu_work(self) -> float:
        return self.current_cpu_burst + sum(self.cpu_bursts[self.current_burst_idx + 1 :])

    @property
    def has_next_io(self) -> bool:
        return self.current_burst_idx < len(self.io_waits)

    def waiting_time(self, now: float) -> float:
        if self.ready_since is None:
            return 0.0
        return max(0.0, now - self.ready_since)

    def mark_ready(self, now: float) -> None:
        self.ready_since = now

    def mark_dispatched(self, now: float) -> None:
        if self.first_started_at is None:
            self.first_started_at = now
        wait_time = self.waiting_time(now)
        self.total_ready_wait_time += wait_time
        self.max_ready_wait_time = max(self.max_ready_wait_time, wait_time)
        self.ready_since = None

    def accumulate_costs(self, energy_cost: float, starvation_cost: float) -> None:
        self.accumulated_energy_cost += energy_cost
        self.accumulated_starvation_cost += starvation_cost

    def finish_current_burst(self, now: float) -> float | None:
        self.cpu_progress += self.current_cpu_burst
        self.current_burst_elapsed = 0.0
        if self.current_burst_idx == len(self.cpu_bursts) - 1:
            self.completed_at = now
            return None

        io_wait = self.io_waits[self.current_burst_idx]
        self.current_burst_idx += 1
        return io_wait

    def preempt_current_burst(self, work_done: float) -> None:
        """Interrupt the current run after `work_done` CPU time was executed.

        The task stays on the same burst (its remaining shrinks) and is meant to
        be returned to the ready queue. `cpu_progress` reflects the partial work.
        """
        work_done = max(0.0, min(work_done, self.current_cpu_burst))
        self.cpu_progress += work_done
        self.current_burst_elapsed += work_done
        self.preemptions += 1

    def response_time(self) -> float | None:
        """Time from arrival until the task first receives CPU service."""
        if self.first_started_at is None:
            return None
        return self.first_started_at - self.arrival_time

    def turnaround_time(self) -> float | None:
        """Time from arrival until full task completion."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.arrival_time
