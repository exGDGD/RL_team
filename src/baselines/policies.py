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
class MLFQPolicy:
    """OS-style multi-level feedback queue baseline.

    This is intentionally not observation-fair: it tracks task identity through
    ``pid`` and uses task runtime history. The simulator has no timer tick, so
    quantum expiry cannot create a new decision point by itself. Instead, queue
    levels are updated whenever the env asks for a scheduling decision.
    """

    levels: int = 3
    quanta: tuple[float, ...] = (4.0, 12.0)
    aging_threshold: float = 30.0
    name: str = "mlfq"

    def reset(self) -> None:
        return None

    def act(self, env: SchedulerEnv, observations: dict[str, dict]) -> dict[str, int]:
        actions = {agent_id: 0 for agent_id in env.agents}
        available = _available_actions(env)

        for agent_id in env.agents:
            if not available:
                break
            core = env.cores[agent_id]
            if core.busy:
                if not env._core_preempt_eligible(core):
                    continue
                best_action = self._best_action(env, available)
                running_level = self._running_level(env, agent_id)
                ready_level = self._effective_level(
                    env.ready_queue[best_action - 1],
                    env.sim.now,
                )
                if ready_level < running_level:
                    actions[agent_id] = best_action
                    available.remove(best_action)
                continue

            best_action = self._best_action(env, available)
            actions[agent_id] = best_action
            available.remove(best_action)

        return actions

    def _best_action(self, env: SchedulerEnv, available: list[int]) -> int:
        now = env.sim.now
        return min(
            available,
            key=lambda action: self._rank(env.ready_queue[action - 1], now),
        )

    def _rank(self, task: Task, now: float) -> tuple[int, float, int]:
        return (
            self._effective_level(task, now),
            task.ready_since if task.ready_since is not None else task.arrival_time,
            task.pid,
        )

    def _effective_level(self, task: Task, now: float) -> int:
        base_level = self._base_level(task.cpu_progress)
        if task.ready_since is None:
            return base_level
        promotions = int(task.waiting_time(now) // max(self.aging_threshold, 1.0e-8))
        return max(0, base_level - promotions)

    def _running_level(self, env: SchedulerEnv, agent_id: str) -> int:
        core = env.cores[agent_id]
        if core.current_task_pid is None:
            return self.levels - 1
        task = env.tasks[core.current_task_pid]
        return self._base_level(task.cpu_progress + self._running_work_done(env, agent_id))

    def _running_work_done(self, env: SchedulerEnv, agent_id: str) -> float:
        core = env.cores[agent_id]
        if core.task_started_at is None or core.current_task_pid is None:
            return 0.0
        elapsed = max(0.0, env.sim.now - core.task_started_at)
        if elapsed <= 0.0:
            return 0.0
        task = env.tasks[core.current_task_pid]
        mismatch = env._mismatch_penalty(core.core_type, task)
        return elapsed * core.spec.speed / max(mismatch, 1.0e-8)

    def _base_level(self, cpu_service: float) -> int:
        level = 0
        remaining_service = cpu_service
        for quantum in self.quanta:
            if remaining_service < quantum:
                return min(level, self.levels - 1)
            remaining_service -= quantum
            level += 1
        return min(level, self.levels - 1)


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
