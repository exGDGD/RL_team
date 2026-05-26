from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.env import CoreType, SchedulerEnv
from src.env.task import LatencyClass, Task


class BaselinePolicy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        ...


def _idle_agents(env: SchedulerEnv) -> list[str]:
    return [agent_id for agent_id in env.agents if not env.cores[agent_id].busy]


def _available_actions(env: SchedulerEnv) -> list[int]:
    return list(range(1, min(env.queue_size, len(env.ready_queue)) + 1))


@dataclass
class RandomPolicy:
    seed: int | None = None
    name: str = "random"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        actions = {agent_id: 0 for agent_id in env.agents}
        available = _available_actions(env)
        for agent_id in _idle_agents(env):
            if not available:
                break
            choice_pos = int(self.rng.integers(0, len(available)))
            actions[agent_id] = available.pop(choice_pos)
        return actions


@dataclass
class RoundRobinPolicy:
    name: str = "round_robin"
    cursor: int = 0

    def reset(self) -> None:
        self.cursor = 0

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        actions = {agent_id: 0 for agent_id in env.agents}
        available = _available_actions(env)
        for agent_id in _idle_agents(env):
            if not available:
                break
            idx = self.cursor % len(available)
            actions[agent_id] = available.pop(idx)
            self.cursor += 1
        return actions


@dataclass
class SJFLikePolicy:
    name: str = "sjf_like"

    def reset(self) -> None:
        return None

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        actions = {agent_id: 0 for agent_id in env.agents}
        available = _available_actions(env)
        for agent_id in _idle_agents(env):
            if not available:
                break
            core = env.cores[agent_id]
            best_action = min(
                available,
                key=lambda action: env._runtime_on_core(core, env.ready_queue[action - 1]),
            )
            actions[agent_id] = best_action
            available.remove(best_action)
        return actions


@dataclass
class EASLikePolicy:
    name: str = "eas_like"

    def reset(self) -> None:
        return None

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        actions = {agent_id: 0 for agent_id in env.agents}
        available = _available_actions(env)
        for agent_id in _idle_agents(env):
            if not available:
                break
            core = env.cores[agent_id]
            best_action = max(
                available,
                key=lambda action: self._score(core.core_type, env.ready_queue[action - 1], env.sim.now),
            )
            actions[agent_id] = best_action
            available.remove(best_action)
        return actions

    def _score(self, core_type: CoreType, task: Task, now: float) -> float:
        wait_bonus = 0.02 * task.waiting_time(now)
        latency_bonus = 2.0 * float(task.latency_class)
        affinity = self._affinity(core_type, task)
        energy_bias = self._energy_bias(core_type, task)
        return affinity + energy_bias + latency_bonus + wait_bonus

    def _affinity(self, core_type: CoreType, task: Task) -> float:
        if task.latency_class == LatencyClass.HARD_RT:
            return {
                CoreType.PRIME: 5.0,
                CoreType.P: 4.0,
                CoreType.E: 1.0,
                CoreType.LP_E: -2.0,
            }[core_type]
        if task.cpu_intensity >= 0.7:
            return {
                CoreType.PRIME: 4.0,
                CoreType.P: 3.5,
                CoreType.E: 1.0,
                CoreType.LP_E: -1.5,
            }[core_type]
        if task.cpu_intensity <= 0.3:
            return {
                CoreType.PRIME: -0.5,
                CoreType.P: 0.0,
                CoreType.E: 3.0,
                CoreType.LP_E: 3.5,
            }[core_type]
        return {
            CoreType.PRIME: 1.0,
            CoreType.P: 2.0,
            CoreType.E: 2.0,
            CoreType.LP_E: 0.5,
        }[core_type]

    def _energy_bias(self, core_type: CoreType, task: Task) -> float:
        if task.latency_class == LatencyClass.BEST_EFFORT and task.cpu_intensity < 0.6:
            return {
                CoreType.PRIME: -1.0,
                CoreType.P: -0.5,
                CoreType.E: 1.0,
                CoreType.LP_E: 1.5,
            }[core_type]
        return 0.0
