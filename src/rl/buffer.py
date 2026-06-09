from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

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
    joint_index: int
    accumulated_reward: float = 0.0


@dataclass(frozen=True)
class AgentTransition:
    """One agent-centric transition for asynchronous actor-critic updates."""

    agent_id: str
    episode_id: int
    agent_index: int
    obs: AgentBatch
    action: int
    log_prob: float
    action_mask: np.ndarray
    reward: float
    next_obs: AgentBatch
    next_agent_index: int
    joint_index: int
    elapsed_time: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class JointMacroTransition:
    """One interval on the shared macro-timeline used by the ACAC critic."""

    episode_id: int
    obs: AgentBatch
    reward: float
    next_obs: AgentBatch
    elapsed_time: float
    terminated: bool
    truncated: bool


@dataclass
class RolloutBuffer:
    transitions: list[AgentTransition] = field(default_factory=list)
    joint_transitions: list[JointMacroTransition] = field(default_factory=list)
    episodes: int = 0
    total_env_reward: float = 0.0
    env_steps: int = 0
    conflicts: int = 0
    invalid_actions: int = 0
    preemptions: int = 0
    decisions: int = 0
    forced_decisions: int = 0
    noop_decisions: int = 0
    total_task_choices: int = 0
    max_task_choices: int = 0

    def append(self, transition: AgentTransition) -> None:
        self.transitions.append(transition)

    def append_joint(self, transition: JointMacroTransition) -> None:
        self.joint_transitions.append(transition)

    def extend(self, other: RolloutBuffer) -> None:
        joint_index_offset = len(self.joint_transitions)
        self.transitions.extend(
            replace(
                transition,
                episode_id=transition.episode_id + self.episodes,
                joint_index=transition.joint_index + joint_index_offset,
            )
            for transition in other.transitions
        )
        self.joint_transitions.extend(
            replace(
                transition,
                episode_id=transition.episode_id + self.episodes,
            )
            for transition in other.joint_transitions
        )
        self.episodes += other.episodes
        self.total_env_reward += other.total_env_reward
        self.env_steps += other.env_steps
        self.conflicts += other.conflicts
        self.invalid_actions += other.invalid_actions
        self.preemptions += other.preemptions
        self.decisions += other.decisions
        self.forced_decisions += other.forced_decisions
        self.noop_decisions += other.noop_decisions
        self.total_task_choices += other.total_task_choices
        self.max_task_choices = max(self.max_task_choices, other.max_task_choices)

    @property
    def mean_task_choices(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.total_task_choices / self.decisions

    @property
    def forced_decision_fraction(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.forced_decisions / self.decisions

    @property
    def noop_fraction(self) -> float:
        """Share of decisions where the agent chose NO-OP (action 0)."""
        if self.decisions == 0:
            return 0.0
        return self.noop_decisions / self.decisions

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


class ReplayBuffer:
    """A small FIFO of recent rollouts reused alongside the current one.

    ACAC is on-policy, so reuse is bounded: only the last ``capacity`` rollouts
    are kept and the PPO clipped ratio (computed against each transition's stored
    ``log_prob``) provides the importance correction for the mild off-policyness.
    ``capacity=0`` disables replay and ``combined`` just returns the current
    rollout, so training behaviour is unchanged by default.
    """

    def __init__(self, capacity: int = 0) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        self.capacity = capacity
        self._buffers: deque[RolloutBuffer] = deque(maxlen=capacity or None)

    def __len__(self) -> int:
        return len(self._buffers)

    def add(self, rollout: RolloutBuffer) -> None:
        """Store a rollout for reuse in later updates (no-op when disabled)."""
        if self.capacity == 0:
            return
        self._buffers.append(rollout)

    def combined(self, current: RolloutBuffer) -> RolloutBuffer:
        """Merge the current rollout with the retained ones into a fresh buffer.

        ``RolloutBuffer.extend`` re-offsets episode and joint indices, so the
        merged buffer keeps every episode's credit assignment self-contained.
        The current rollout is appended first to preserve its diagnostics order.
        """
        merged = RolloutBuffer()
        merged.extend(current)
        for past in self._buffers:
            merged.extend(past)
        return merged
