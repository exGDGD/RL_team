from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.env import CoreType


@dataclass(frozen=True)
class AgentBatch:
    """A stable numpy contract between SchedulerEnv and RL code.

    SchedulerEnv observations are convenient dictionaries. Policy and rollout
    code need aligned arrays, explicit decision masks, and type grouping for
    shared actors. This batch keeps those details in one place.
    """

    agent_ids: tuple[str, ...]
    core_type_indices: np.ndarray
    self_features: np.ndarray
    ready_queue: np.ndarray
    ready_mask: np.ndarray
    other_cores: np.ndarray
    other_core_mask: np.ndarray
    system: np.ndarray
    action_mask: np.ndarray
    delta_t: np.ndarray
    decision_mask: np.ndarray

    @property
    def num_agents(self) -> int:
        return len(self.agent_ids)

    @property
    def queue_size(self) -> int:
        return int(self.ready_queue.shape[1])

    def indices_for_core_type(self, core_type: CoreType) -> np.ndarray:
        return np.flatnonzero(self.core_type_indices == _core_type_index(core_type))

    def decision_agent_ids(self) -> tuple[str, ...]:
        return tuple(
            agent_id
            for agent_id, should_act in zip(self.agent_ids, self.decision_mask, strict=True)
            if bool(should_act)
        )


def build_agent_batch(
    observations: dict[str, dict[str, Any]],
    *,
    agent_order: list[str] | tuple[str, ...] | None = None,
) -> AgentBatch:
    """Convert env observations into aligned arrays for rollout collection."""

    if agent_order is None:
        agent_ids = tuple(observations.keys())
    else:
        agent_ids = tuple(agent_order)

    if not agent_ids:
        raise ValueError("Cannot build AgentBatch from empty observations.")

    missing = [agent_id for agent_id in agent_ids if agent_id not in observations]
    if missing:
        raise KeyError(f"Missing observations for agents: {missing}")

    self_features = _stack(agent_ids, observations, "self", dtype=np.float32)
    ready_queue = _stack(agent_ids, observations, "ready_queue", dtype=np.float32)
    ready_mask = _stack(agent_ids, observations, "ready_mask", dtype=np.int8)
    other_cores = _stack(agent_ids, observations, "other_cores", dtype=np.float32)
    system = _stack(agent_ids, observations, "system", dtype=np.float32)
    action_mask = _stack(agent_ids, observations, "action_mask", dtype=np.int8)

    core_type_indices = self_features[:, 0].astype(np.int64)
    delta_t = self_features[:, 4].astype(np.float32)
    is_idle = self_features[:, 1] == 0.0
    has_task_action = np.any(action_mask[:, 1:] == 1, axis=1)
    decision_mask = np.logical_and(is_idle, has_task_action)
    other_core_mask = np.ones(other_cores.shape[:2], dtype=np.int8)

    return AgentBatch(
        agent_ids=agent_ids,
        core_type_indices=core_type_indices,
        self_features=self_features,
        ready_queue=ready_queue,
        ready_mask=ready_mask,
        other_cores=other_cores,
        other_core_mask=other_core_mask,
        system=system,
        action_mask=action_mask,
        delta_t=delta_t,
        decision_mask=decision_mask,
    )


def agents_by_core_type(batch: AgentBatch) -> dict[CoreType, np.ndarray]:
    """Return batch row indices for each shared actor group."""

    return {
        core_type: batch.indices_for_core_type(core_type)
        for core_type in CoreType
    }


def mask_logits(
    logits: np.ndarray,
    action_mask: np.ndarray,
    *,
    invalid_value: float = -1.0e9,
) -> np.ndarray:
    """Apply invalid-action masking to action logits.

    Args:
        logits: Array shaped (..., action_dim).
        action_mask: Binary array broadcastable to logits where 1 means valid.
        invalid_value: Value written into invalid entries before sampling.
    """

    logits = np.asarray(logits, dtype=np.float32)
    action_mask = np.asarray(action_mask)
    if logits.shape != action_mask.shape:
        try:
            action_mask = np.broadcast_to(action_mask, logits.shape)
        except ValueError as exc:
            raise ValueError(
                f"action_mask shape {action_mask.shape} is not broadcastable "
                f"to logits shape {logits.shape}"
            ) from exc

    return np.where(action_mask.astype(bool), logits, invalid_value).astype(np.float32)


def _stack(
    agent_ids: tuple[str, ...],
    observations: dict[str, dict[str, Any]],
    key: str,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    return np.stack(
        [np.asarray(observations[agent_id][key], dtype=dtype) for agent_id in agent_ids],
        axis=0,
    )


def _core_type_index(core_type: CoreType) -> int:
    return list(CoreType).index(core_type)
