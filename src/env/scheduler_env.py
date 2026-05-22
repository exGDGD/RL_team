from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import simpy
from gymnasium import spaces

from .core import Core, CoreType
from .metrics import EpisodeMetrics, compute_episode_metrics
from .spaces import build_action_space, build_observation_space
from .task import LatencyClass, Task
from .workload import WorkloadConfig, WorkloadGenerator, WorkloadScenario


DEFAULT_CORE_CONFIG = {
    CoreType.P: 2,
    CoreType.E: 2,
}


@dataclass(frozen=True)
class RewardWeights:
    completion: float = 10.0
    energy: float = 0.1
    starvation: float = 0.001
    latency: float = 0.05


class RewardMode(str, Enum):
    EVENT_COST = "event_cost"
    COMPLETION_ONLY = "completion_only"


class SchedulerEnv:
    """First-pass event-driven CPU scheduling environment.

    The public API intentionally resembles PettingZoo's ParallelEnv shape:
    actions and observations are dictionaries keyed by core/agent id.

    Semantics fixed in this simulator:
    - An action selects from the ready-queue snapshot visible at step start.
    - If multiple idle cores select the same task, the earliest agent in
      deterministic agent order wins and later duplicate claims become NO-OP.
    - RewardMode.EVENT_COST emits energy/starvation cost at each CPU burst and
      completion/latency terms when the task finishes.
    - RewardMode.COMPLETION_ONLY accumulates burst costs internally and emits
      all reward terms only when the task finishes.
    - episode_time is the workload arrival horizon. Episodes terminate after the
      generated finite trace drains, or truncate at max_sim_time as a guard.
    """

    metadata = {"name": "heterogeneous_cpu_scheduler_v0"}

    def __init__(
        self,
        core_config: dict[CoreType, int] | None = None,
        queue_size: int = 8,
        workload_scenario: WorkloadScenario = WorkloadScenario.BALANCED,
        arrival_rate: float = 0.15,
        episode_time: float = 500.0,
        max_sim_time: float | None = None,
        max_tasks: int = 64,
        seed: int | None = None,
        reward_weights: RewardWeights | None = None,
        reward_mode: RewardMode | str = RewardMode.EVENT_COST,
    ) -> None:
        self.core_config = core_config or DEFAULT_CORE_CONFIG
        self.queue_size = queue_size
        self.workload_config = WorkloadConfig(
            scenario=workload_scenario,
            arrival_rate=arrival_rate,
            episode_time=episode_time,
            max_tasks=max_tasks,
        )
        self.seed = seed
        self.reward_weights = reward_weights or RewardWeights()
        self.reward_mode = RewardMode(reward_mode)
        self.max_sim_time = max_sim_time if max_sim_time is not None else episode_time * 10.0
        self._action_space = build_action_space(queue_size=self.queue_size)
        self._observation_space = build_observation_space(
            queue_size=self.queue_size,
            num_cores=sum(self.core_config.values()),
        )

        self.sim: simpy.Environment
        self.cores: dict[str, Core]
        self.agents: list[str]
        self.tasks: dict[int, Task]
        self.ready_queue: list[Task]
        self._arrival_events: list[tuple[float, int]]
        self._running_events: list[tuple[float, int, str, int, float]]
        self._io_events: list[tuple[float, int]]
        self._event_seq: int
        self._last_rewards: dict[str, float]
        self._last_step_info: dict[str, Any]
        self.completed_tasks: list[Task]

    def action_space(self, agent_id: str | None = None) -> spaces.Discrete:
        self._validate_agent_id(agent_id)
        return self._action_space

    def observation_space(self, agent_id: str | None = None) -> spaces.Dict:
        self._validate_agent_id(agent_id)
        return self._observation_space

    def reset(self, seed: int | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if seed is not None:
            self.seed = seed

        self.sim = simpy.Environment()
        self.cores = self._build_cores()
        self.agents = list(self.cores.keys())
        generator = WorkloadGenerator(self.workload_config, seed=self.seed)
        generated_tasks = generator.generate()
        self.tasks = {task.pid: task for task in generated_tasks}
        self.ready_queue = []
        self.completed_tasks = []
        self._event_seq = 0
        self._arrival_events = [(task.arrival_time, task.pid) for task in generated_tasks]
        heapq.heapify(self._arrival_events)
        self._running_events = []
        self._io_events = []
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._last_step_info = {
            "assignments": {},
            "conflicts": {},
            "invalid_actions": {},
        }

        self._advance_to_decision()
        return self.observe(), self.info()

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        self._last_rewards = {agent: 0.0 for agent in self.agents}
        self._last_step_info = self._resolve_and_dispatch(actions)

        self._advance_to_decision()
        terminated = self._terminated()
        truncated = (not terminated) and self._truncated()
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        return self.observe(), dict(self._last_rewards), terminations, truncations, self.info()

    def observe(self) -> dict[str, dict[str, Any]]:
        return {agent_id: self._observe_agent(agent_id) for agent_id in self.agents}

    def info(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "time": self.sim.now,
            "ready_queue_len": len(self.ready_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_tasks": len(self.tasks),
            "mean_response_time": self._mean_response_time(),
            "metrics": metrics.as_dict(),
            "assignments": dict(self._last_step_info["assignments"]),
            "conflicts": dict(self._last_step_info["conflicts"]),
            "invalid_actions": dict(self._last_step_info["invalid_actions"]),
        }

    def metrics(self, starvation_threshold: float = 100.0) -> EpisodeMetrics:
        return compute_episode_metrics(
            tasks=self.tasks.values(),
            cores=self.cores,
            now=self.sim.now,
            starvation_threshold=starvation_threshold,
        )

    def _build_cores(self) -> dict[str, Core]:
        cores: dict[str, Core] = {}
        for core_type, count in self.core_config.items():
            for idx in range(count):
                core_id = f"{core_type.value}_{idx}"
                cores[core_id] = Core(core_id=core_id, core_type=core_type)
        return cores

    def _validate_agent_id(self, agent_id: str | None) -> None:
        if agent_id is None:
            return
        expected_agents = self.agents if hasattr(self, "agents") else list(self._build_cores())
        if agent_id not in expected_agents:
            raise KeyError(f"Unknown agent id: {agent_id}")

    def _advance_to_decision(self) -> None:
        while not self._episode_over():
            self._release_arrivals()
            self._release_io()
            if self.ready_queue and any(not core.busy for core in self.cores.values()):
                return

            next_time = self._next_event_time()
            if next_time is None:
                return
            if next_time > self.max_sim_time:
                self.sim.run(until=self.max_sim_time)
                return
            if next_time > self.sim.now:
                self.sim.run(until=next_time)

            self._complete_finished_runs()

    def _resolve_and_dispatch(self, actions: dict[str, int]) -> dict[str, Any]:
        queue_snapshot = list(self.ready_queue[: self.queue_size])
        claimed_pids: set[int] = set()
        assignments: dict[str, int] = {}
        conflicts: dict[str, int] = {}
        invalid_actions: dict[str, int] = {}

        for agent_id in self.agents:
            core = self.cores[agent_id]
            action = int(actions.get(agent_id, 0))
            if core.busy or action == 0:
                continue
            if action < 0 or action > self.queue_size:
                invalid_actions[agent_id] = action
                continue

            queue_idx = action - 1
            if queue_idx >= len(queue_snapshot):
                invalid_actions[agent_id] = action
                continue

            task = queue_snapshot[queue_idx]
            if task.pid in claimed_pids:
                conflicts[agent_id] = task.pid
                continue

            live_task = self._pop_ready_task(task.pid)
            if live_task is None:
                conflicts[agent_id] = task.pid
                continue

            claimed_pids.add(task.pid)
            assignments[agent_id] = task.pid
            self._dispatch(core, live_task)

        return {
            "assignments": assignments,
            "conflicts": conflicts,
            "invalid_actions": invalid_actions,
        }

    def _pop_ready_task(self, pid: int) -> Task | None:
        for idx, task in enumerate(self.ready_queue):
            if task.pid == pid:
                return self.ready_queue.pop(idx)
        return None

    def _next_event_time(self) -> float | None:
        candidates = []
        if self._arrival_events:
            candidates.append(self._arrival_events[0][0])
        if self._running_events:
            candidates.append(self._running_events[0][0])
        if self._io_events:
            candidates.append(self._io_events[0][0])
        return min(candidates) if candidates else None

    def _release_arrivals(self) -> None:
        while self._arrival_events and self._arrival_events[0][0] <= self.sim.now:
            _, pid = heapq.heappop(self._arrival_events)
            task = self.tasks[pid]
            task.mark_ready(self.sim.now)
            self.ready_queue.append(task)

    def _release_io(self) -> None:
        while self._io_events and self._io_events[0][0] <= self.sim.now:
            _, pid = heapq.heappop(self._io_events)
            task = self.tasks[pid]
            task.mark_ready(self.sim.now)
            self.ready_queue.append(task)

    def _dispatch(self, core: Core, task: Task) -> None:
        task.mark_dispatched(self.sim.now)
        core.assign(task.pid, self.sim.now)
        run_time = self._runtime_on_core(core, task)
        self._event_seq += 1
        heapq.heappush(
            self._running_events,
            (self.sim.now + run_time, self._event_seq, core.core_id, task.pid, run_time),
        )

    def _complete_finished_runs(self) -> None:
        while self._running_events and self._running_events[0][0] <= self.sim.now:
            _, _, core_id, pid, run_time = heapq.heappop(self._running_events)
            core = self.cores[core_id]
            task = self.tasks[pid]
            core.release(self.sim.now, run_time)

            energy_cost, starvation_cost = self._burst_costs(core, run_time)
            task.accumulate_costs(
                energy_cost=energy_cost,
                starvation_cost=starvation_cost,
            )
            io_wait = task.finish_current_burst(self.sim.now)
            reward = self._reward_for_finished_burst(
                task=task,
                energy_cost=energy_cost,
                starvation_cost=starvation_cost,
            )
            self._last_rewards[core_id] += reward
            if task.done:
                self.completed_tasks.append(task)
            elif io_wait is not None:
                heapq.heappush(self._io_events, (self.sim.now + io_wait, task.pid))

    def _runtime_on_core(self, core: Core, task: Task) -> float:
        mismatch = self._mismatch_penalty(core.core_type, task)
        return task.current_cpu_burst / core.spec.speed * mismatch

    def _mismatch_penalty(self, core_type: CoreType, task: Task) -> float:
        if task.latency_class == LatencyClass.HARD_RT and core_type in {CoreType.E, CoreType.LP_E}:
            return 1.4
        if task.cpu_intensity > 0.75 and core_type == CoreType.LP_E:
            return 1.5
        if task.cpu_intensity < 0.25 and core_type in {CoreType.PRIME, CoreType.P}:
            return 1.15
        return 1.0

    def _burst_costs(self, core: Core, run_time: float) -> tuple[float, float]:
        energy_cost = core.spec.power * run_time
        starvation_cost = sum(t.waiting_time(self.sim.now) ** 2 for t in self.ready_queue) * run_time
        return energy_cost, starvation_cost

    def _reward_for_finished_burst(
        self,
        task: Task,
        energy_cost: float,
        starvation_cost: float,
    ) -> float:
        weights = self.reward_weights
        latency_cost = 0.0
        if task.done and task.response_time() is not None:
            latency_cost = float(task.latency_class) * task.response_time()

        if self.reward_mode == RewardMode.EVENT_COST:
            reward = -weights.energy * energy_cost - weights.starvation * starvation_cost
            if task.done:
                reward += weights.completion - weights.latency * latency_cost
            return reward

        if not task.done:
            return 0.0

        return (
            weights.completion
            - weights.energy * task.accumulated_energy_cost
            - weights.starvation * task.accumulated_starvation_cost
            - weights.latency * latency_cost
        )

    def _observe_agent(self, agent_id: str) -> dict[str, Any]:
        core = self.cores[agent_id]
        now = self.sim.now
        ready_features = np.zeros((self.queue_size, 4), dtype=np.float32)
        ready_mask = np.zeros((self.queue_size,), dtype=np.int8)
        for idx, task in enumerate(self.ready_queue[: self.queue_size]):
            ready_features[idx] = np.array(
                [
                    task.waiting_time(now),
                    task.cpu_progress,
                    float(task.latency_class),
                    task.cpu_intensity,
                ],
                dtype=np.float32,
            )
            ready_mask[idx] = 1

        other_core_features = []
        for other_id, other in self.cores.items():
            if other_id == agent_id:
                continue
            elapsed = 0.0 if other.task_started_at is None else now - other.task_started_at
            other_core_features.append(
                [
                    self._core_type_index(other.core_type),
                    float(other.busy),
                    elapsed,
                ]
            )

        type_counts = np.array(
            [self.core_config.get(core_type, 0) for core_type in CoreType],
            dtype=np.float32,
        )
        utilization = (
            sum(float(c.busy) for c in self.cores.values()) / max(1, len(self.cores))
        )
        elapsed_current = 0.0 if core.task_started_at is None else now - core.task_started_at
        return {
            "self": np.array(
                [
                    self._core_type_index(core.core_type),
                    float(core.busy),
                    elapsed_current,
                    core.accumulated_energy,
                    now - core.last_decision_time,
                ],
                dtype=np.float32,
            ),
            "ready_queue": ready_features,
            "ready_mask": ready_mask,
            "other_cores": np.array(other_core_features, dtype=np.float32),
            "system": np.concatenate(
                [
                    np.array([len(self.cores), utilization], dtype=np.float32),
                    type_counts,
                ]
            ),
            "action_mask": np.concatenate(
                [np.array([1], dtype=np.int8), ready_mask]
            ),
        }

    def _core_type_index(self, core_type: CoreType) -> int:
        return list(CoreType).index(core_type)

    def _episode_over(self) -> bool:
        return self._terminated() or self._truncated()

    def _terminated(self) -> bool:
        no_pending_events = not (self._arrival_events or self._running_events or self._io_events)
        return no_pending_events and not self.ready_queue

    def _truncated(self) -> bool:
        return self.sim.now >= self.max_sim_time

    def _mean_response_time(self) -> float | None:
        response_times = [task.response_time() for task in self.completed_tasks]
        valid = [value for value in response_times if value is not None]
        if not valid:
            return None
        return float(np.mean(valid))
