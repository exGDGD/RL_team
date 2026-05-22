from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoreType(str, Enum):
    PRIME = "prime"
    P = "p"
    E = "e"
    LP_E = "lp_e"


@dataclass(frozen=True)
class CoreSpec:
    speed: float
    power: float
    context_switch_cost: float


CORE_SPECS: dict[CoreType, CoreSpec] = {
    CoreType.PRIME: CoreSpec(speed=4.0, power=8.0, context_switch_cost=1.5),
    CoreType.P: CoreSpec(speed=3.0, power=5.0, context_switch_cost=1.0),
    CoreType.E: CoreSpec(speed=1.5, power=1.5, context_switch_cost=0.3),
    CoreType.LP_E: CoreSpec(speed=0.8, power=0.4, context_switch_cost=0.2),
}


@dataclass
class Core:
    core_id: str
    core_type: CoreType
    current_task_pid: int | None = None
    task_started_at: float | None = None
    accumulated_energy: float = 0.0
    busy_time: float = 0.0
    last_decision_time: float = 0.0

    @property
    def spec(self) -> CoreSpec:
        return CORE_SPECS[self.core_type]

    @property
    def busy(self) -> bool:
        return self.current_task_pid is not None

    def assign(self, task_pid: int, now: float) -> None:
        self.current_task_pid = task_pid
        self.task_started_at = now
        self.last_decision_time = now

    def release(self, now: float, run_time: float) -> None:
        self.accumulated_energy += self.spec.power * run_time
        self.busy_time += run_time
        self.current_task_pid = None
        self.task_started_at = None
        self.last_decision_time = now
