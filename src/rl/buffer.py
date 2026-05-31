from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .obs import AgentBatch


@dataclass(frozen=True)
class PendingDecision:
    """An action selected by one agent whose reward has not arrived yet."""

    agent_id: str
    agent_index: int
    obs: AgentBatch
    action: int
    log_prob: float
    action_mask: np.ndarray
    start_time: float


@dataclass(frozen=True)
class AgentTransition:
    """One agent-centric transition for asynchronous actor-critic updates."""

    agent_id: str
    agent_index: int
    obs: AgentBatch
    action: int
    log_prob: float
    action_mask: np.ndarray
    reward: float
    next_obs: AgentBatch
    next_agent_index: int
    elapsed_time: float
    terminated: bool
    truncated: bool


@dataclass
class RolloutBuffer:
    transitions: list[AgentTransition] = field(default_factory=list)
    env_steps: int = 0
    conflicts: int = 0
    invalid_actions: int = 0

    def append(self, transition: AgentTransition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


def compute_time_scaled_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    delta_t: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ACAC-style GAE using wall-clock time discounts.

    Arrays are shaped (T,) for one agent trajectory. `delta_t[t]` is the elapsed
    simulated time associated with transition t. Terminal transitions zero out
    bootstrap and recursive advantage terms.
    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    delta_t = np.asarray(delta_t, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)

    _ensure_same_shape(rewards, values, next_values, delta_t, dones)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.float32(0.0)
    for idx in range(len(rewards) - 1, -1, -1):
        not_done = np.float32(0.0 if dones[idx] else 1.0)
        discount = np.float32(gamma ** float(delta_t[idx]))
        delta = rewards[idx] + discount * next_values[idx] * not_done - values[idx]
        last_gae = delta + discount * np.float32(gae_lambda) * not_done * last_gae
        advantages[idx] = last_gae

    returns = advantages + values
    return advantages, returns.astype(np.float32)


def _ensure_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All arrays must have the same shape, got {sorted(shapes)}")
