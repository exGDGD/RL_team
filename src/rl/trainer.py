from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.env import CoreType

from .buffer import (
    AgentTransition,
    JointMacroTransition,
    RolloutBuffer,
    compute_time_scaled_gae,
)
from .networks import AgentCentricCritic, TypeSharedActor, mask_logits
from .obs import AgentBatch


@dataclass(frozen=True)
class ACACConfig:
    hidden_dim: int = 128
    critic_heads: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.05
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    allow_noop: bool = True
    update_epochs: int = 2
    num_minibatches: int = 1
    reward_scale: float = 0.01
    # "global": standardize advantages across the whole merged rollout (the
    # largest-magnitude scenario then dominates the policy gradient).
    # "per_episode": standardize within each rollout episode so every scenario
    # contributes an equal-scale gradient regardless of its raw reward scale.
    advantage_norm: str = "global"


@dataclass(frozen=True)
class UpdateStats:
    actor_samples: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    actor_grad_norm: float
    critic_grad_norm: float
    advantage_mean: float
    advantage_std: float
    return_mean: float
    normalized_entropy: float
    ratio_std: float
    ratio_max_deviation: float
    entropy_coef: float = 0.0


class TorchACACPolicy(nn.Module):
    """Type-shared actor set plus one centralized agent-centric critic."""

    def __init__(
        self,
        config: ACACConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or ACACConfig()
        self.device = torch.device(device)
        self.actors = nn.ModuleDict(
            {
                core_type.value: TypeSharedActor(hidden_dim=self.config.hidden_dim)
                for core_type in CoreType
            }
        )
        self.critic = AgentCentricCritic(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.critic_heads,
        )
        self.to(self.device)
        self._actor_hidden: dict[str, torch.Tensor] = {}

    def reset_recurrent_state(
        self,
        agent_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Reset rollout-time actor memory.

        The trainer never uses this mutable state during PPO updates; updates
        replay the hidden states stored on each transition. This state is only
        for live action selection during rollout/evaluation.
        """

        if agent_ids is None:
            self._actor_hidden.clear()
            return
        for agent_id in agent_ids:
            self._actor_hidden[agent_id] = torch.zeros(
                self.config.hidden_dim,
                dtype=torch.float32,
                device=self.device,
            )

    def act(
        self,
        batch: AgentBatch,
        *,
        deterministic: bool = False,
    ) -> tuple[
        dict[str, int],
        dict[str, float],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        actions = {agent_id: 0 for agent_id in batch.agent_ids}
        log_probs = {agent_id: 0.0 for agent_id in batch.agent_ids}
        effective_masks: dict[str, np.ndarray] = {}
        actor_hiddens: dict[str, np.ndarray] = {}
        claimed_slots: set[int] = set()

        decision_rows = [
            row
            for row in range(len(batch.agent_ids))
            if bool(batch.decision_mask[row])
        ]
        decision_row_set = set(decision_rows)

        with torch.no_grad():
            # Forward pass batched per core type (one call per type instead of
            # one per core). The action logits do not depend on claimed_slots,
            # so they are computed up front; only the cheap mask+sample step
            # below stays sequential to preserve the exact claimed-slot logic
            # and RNG draw order of the unbatched implementation.
            raw_logits: dict[int, torch.Tensor] = {}
            base_masks: dict[int, torch.Tensor] = {}
            next_hidden_by_row: dict[int, torch.Tensor] = {}
            rows_by_type: dict[CoreType, list[int]] = {}
            for row in range(len(batch.agent_ids)):
                core_type = list(CoreType)[int(batch.core_type_indices[row])]
                rows_by_type.setdefault(core_type, []).append(row)
            for core_type, rows in rows_by_type.items():
                tensors = batch_rows_to_tensors(batch, rows, self.device)
                tensors = self._apply_policy_action_mask(tensors)
                actor_inputs = _actor_inputs(tensors)
                actor_inputs["action_mask"] = None
                hidden = torch.stack(
                    [
                        self._actor_hidden.get(
                            batch.agent_ids[row],
                            torch.zeros(
                                self.config.hidden_dim,
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        )
                        for row in rows
                    ]
                )
                for offset, row in enumerate(rows):
                    if row in decision_row_set:
                        actor_hiddens[batch.agent_ids[row]] = (
                            hidden[offset].detach().cpu().numpy().astype(np.float32)
                        )
                logits, next_hidden = self.actors[core_type.value](
                    **actor_inputs,
                    actor_hidden=hidden,
                    return_hidden=True,
                )
                for offset, row in enumerate(rows):
                    next_hidden_by_row[row] = next_hidden[offset].detach()
                    if row in decision_row_set:
                        raw_logits[row] = logits[offset]
                        base_masks[row] = tensors["action_mask"][offset]

            for row, next_hidden in next_hidden_by_row.items():
                self._actor_hidden[batch.agent_ids[row]] = next_hidden

            for row in decision_rows:
                agent_id = batch.agent_ids[row]
                mask = base_masks[row].clone()
                for claimed_slot in claimed_slots:
                    mask[claimed_slot] = False
                if not torch.any(mask):
                    continue
                masked_logits = mask_logits(
                    raw_logits[row].unsqueeze(0),
                    mask.unsqueeze(0),
                )
                dist = Categorical(logits=masked_logits)
                sampled_actions = (
                    torch.argmax(masked_logits, dim=-1)
                    if deterministic
                    else dist.sample()
                )
                sampled_log_probs = dist.log_prob(sampled_actions)

                action = int(sampled_actions[0].item())
                actions[agent_id] = action
                log_probs[agent_id] = float(sampled_log_probs[0].item())
                effective_masks[agent_id] = mask.cpu().numpy()
                if action > 0:
                    claimed_slots.add(action)

        return actions, log_probs, effective_masks, actor_hiddens

    def evaluate_transitions(
        self,
        transitions: list[AgentTransition],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not transitions:
            empty = torch.empty(0, device=self.device)
            return empty, empty

        log_probs: list[torch.Tensor | None] = [None] * len(transitions)
        entropies: list[torch.Tensor | None] = [None] * len(transitions)

        # Group transitions by core type so each type-shared actor runs a single
        # batched forward instead of one forward per transition.
        groups: dict[CoreType, list[int]] = {}
        for position, transition in enumerate(transitions):
            core_type = list(CoreType)[
                int(transition.obs.core_type_indices[transition.agent_index])
            ]
            groups.setdefault(core_type, []).append(position)

        for core_type, positions in groups.items():
            rows = []
            hidden_rows = []
            for position in positions:
                transition = transitions[position]
                raw = _raw_rows_to_tensors(
                    transition.obs, [transition.agent_index], self.device
                )
                # Use the stored effective mask (post claimed-slot), not the raw
                # batch mask, to match the action that was actually sampled.
                raw["action_mask"] = torch.as_tensor(
                    transition.action_mask,
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0)
                rows.append(raw)
                hidden_rows.append(
                    _transition_actor_hidden(
                        transition,
                        hidden_dim=self.config.hidden_dim,
                        device=self.device,
                    )
                )
            batched = normalize_observation_tensors(_stack_tensor_dicts(rows))
            logits = self.actors[core_type.value](
                **_actor_inputs(batched),
                actor_hidden=torch.stack(hidden_rows),
            )
            dist = Categorical(logits=logits)
            chosen = torch.tensor(
                [transitions[position].action for position in positions],
                dtype=torch.long,
                device=self.device,
            )
            group_log_probs = dist.log_prob(chosen)
            group_entropies = dist.entropy()
            for offset, position in enumerate(positions):
                log_probs[position] = group_log_probs[offset]
                entropies[position] = group_entropies[offset]

        return torch.stack(log_probs), torch.stack(entropies)

    def imitation_logits(
        self,
        *,
        batch: AgentBatch,
        agent_index: int,
        action_mask: np.ndarray,
    ) -> torch.Tensor:
        tensors = batch_rows_to_tensors(batch, [agent_index], self.device)
        tensors["action_mask"] = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        core_type = list(CoreType)[int(batch.core_type_indices[agent_index])]
        return self.actors[core_type.value](**_actor_inputs(tensors)).squeeze(0)

    def _apply_policy_action_mask(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.config.allow_noop:
            return tensors
        tensors = dict(tensors)
        tensors["action_mask"] = tensors["action_mask"].clone()
        tensors["action_mask"][:, 0] = False
        return tensors

    def _critic_values(self, batches: list[AgentBatch]) -> torch.Tensor:
        """Per-batch mean agent value, computed with a single batched critic
        forward per agent-count group instead of one forward per transition."""

        if not batches:
            return torch.empty(0, device=self.device)

        values: list[torch.Tensor | None] = [None] * len(batches)
        # Group by agent count so rows stack cleanly (the per-agent ``other``
        # dimension equals num_agents - 1, which must match within a batch).
        groups: dict[int, list[int]] = {}
        for position, batch in enumerate(batches):
            groups.setdefault(batch.num_agents, []).append(position)

        for _, positions in groups.items():
            rows = []
            counts = []
            for position in positions:
                batch = batches[position]
                rows.append(
                    _raw_rows_to_tensors(
                        batch, list(range(batch.num_agents)), self.device
                    )
                )
                counts.append(batch.num_agents)
            batched = normalize_observation_tensors(_stack_tensor_dicts(rows))
            flat_values = self.critic(**_critic_inputs(batched))
            start = 0
            for offset, position in enumerate(positions):
                count = counts[offset]
                values[position] = flat_values[start : start + count].mean()
                start += count

        return torch.stack(values)

    def values_for_joint_transitions(
        self,
        transitions: list[JointMacroTransition],
        *,
        next_obs: bool = False,
    ) -> torch.Tensor:
        batches = [
            transition.next_obs if next_obs else transition.obs
            for transition in transitions
        ]
        with torch.no_grad():
            return self._critic_values(batches)

    def evaluate_joint_transitions(
        self,
        transitions: list[JointMacroTransition],
    ) -> torch.Tensor:
        return self._critic_values([transition.obs for transition in transitions])


class ACACTrainer:
    def __init__(
        self,
        policy: TorchACACPolicy,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.policy = policy
        self.config = policy.config
        self.optimizer = optimizer or torch.optim.Adam(
            [
                {
                    "params": policy.actors.parameters(),
                    "lr": self.config.actor_learning_rate,
                },
                {
                    "params": policy.critic.parameters(),
                    "lr": self.config.critic_learning_rate,
                },
            ]
        )
        # Dedicated CPU generator so minibatch shuffling never perturbs the
        # global torch RNG stream that the rollout's action sampling draws from
        # (keeps seeded runs reproducible; full-batch updates are order-agnostic).
        self._shuffle_rng = torch.Generator()
        self._shuffle_rng.manual_seed(0)

    def update(
        self,
        rollout: RolloutBuffer,
        *,
        entropy_coef: float | None = None,
    ) -> UpdateStats:
        """Run ``update_epochs`` passes over the rollout.

        ``entropy_coef`` overrides ``config.entropy_coef`` for this update (the
        training loop passes the annealed value); ``None`` keeps the config one.
        With ``config.num_minibatches > 1`` each epoch is split into minibatch
        SGD steps over the macro-timeline intervals (each minibatch carries both
        its critic targets and the actor decisions credited to those intervals).
        """

        transitions = rollout.transitions
        actor_transitions = [
            transition
            for transition in transitions
            if np.count_nonzero(transition.action_mask) > 1
        ]
        joint_transitions = rollout.joint_transitions
        if not transitions or not joint_transitions:
            raise ValueError("Cannot update from an empty rollout buffer.")
        if not actor_transitions:
            raise ValueError("Rollout has no learnable actor transitions.")

        coef = (
            self.config.entropy_coef
            if entropy_coef is None
            else float(entropy_coef)
        )

        with torch.no_grad():
            old_values = self.policy.values_for_joint_transitions(joint_transitions)
            next_values = self.policy.values_for_joint_transitions(
                joint_transitions,
                next_obs=True,
            )
            joint_advantages, joint_returns = compute_joint_advantages(
                transitions=joint_transitions,
                values=old_values.cpu().numpy(),
                next_values=next_values.cpu().numpy(),
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                reward_scale=self.config.reward_scale,
            )
            advantages = map_actor_advantages(
                transitions=actor_transitions,
                joint_advantages=joint_advantages,
            )

        old_log_probs = torch.tensor(
            [transition.log_prob for transition in actor_transitions],
            dtype=torch.float32,
            device=self.policy.device,
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.policy.device)
        returns_t = torch.tensor(joint_returns, dtype=torch.float32, device=self.policy.device)
        if self.config.advantage_norm == "global":
            advantages_t = normalize_advantages(advantages_t)
        else:
            advantages_t = normalize_advantages_by_group(
                advantages_t,
                advantage_group_ids(actor_transitions, rollout, self.config.advantage_norm),
            )

        # Each learnable actor transition is credited to the macro-interval it
        # started in (``joint_index``); grouping lets a minibatch of intervals
        # gather its actor decisions in one indexable list.
        actor_by_joint: dict[int, list[int]] = {}
        for position, transition in enumerate(actor_transitions):
            actor_by_joint.setdefault(transition.joint_index, []).append(position)

        num_joint = len(joint_transitions)
        num_minibatches = max(1, min(self.config.num_minibatches, num_joint))

        step_stats: list[tuple[float, ...]] = []
        for _ in range(self.config.update_epochs):
            perm = torch.randperm(num_joint, generator=self._shuffle_rng).tolist()
            for joint_idx in _chunk(perm, num_minibatches):
                actor_idx = [
                    position
                    for joint_position in joint_idx
                    for position in actor_by_joint.get(joint_position, [])
                ]
                stat = self._minibatch_step(
                    actor_transitions=actor_transitions,
                    joint_transitions=joint_transitions,
                    actor_idx=actor_idx,
                    joint_idx=joint_idx,
                    old_log_probs=old_log_probs,
                    advantages_t=advantages_t,
                    returns_t=returns_t,
                    entropy_coef=coef,
                )
                if stat is not None:
                    step_stats.append(stat)

        means = np.mean(step_stats, axis=0)
        return UpdateStats(
            actor_samples=len(actor_transitions),
            loss=float(means[0]),
            policy_loss=float(means[1]),
            value_loss=float(means[2]),
            entropy=float(means[3]),
            approx_kl=float(means[4]),
            clip_fraction=float(means[5]),
            actor_grad_norm=float(means[6]),
            critic_grad_norm=float(means[7]),
            advantage_mean=float(np.mean(advantages)),
            advantage_std=float(np.std(advantages)),
            return_mean=float(np.mean(joint_returns)),
            normalized_entropy=float(means[8]),
            ratio_std=float(means[9]),
            ratio_max_deviation=float(means[10]),
            entropy_coef=float(coef),
        )

    def _minibatch_step(
        self,
        *,
        actor_transitions: list[AgentTransition],
        joint_transitions: list[JointMacroTransition],
        actor_idx: list[int],
        joint_idx: list[int],
        old_log_probs: torch.Tensor,
        advantages_t: torch.Tensor,
        returns_t: torch.Tensor,
        entropy_coef: float,
    ) -> tuple[float, ...] | None:
        """One optimizer step over a minibatch of macro-intervals.

        Returns the per-step diagnostics, or ``None`` for a value-only minibatch
        (no learnable actor decisions in these intervals) which still updates the
        critic but has no policy statistics to report.
        """

        mb_joint = [joint_transitions[j] for j in joint_idx]
        values = self.policy.evaluate_joint_transitions(mb_joint)
        mb_returns = returns_t[joint_idx]
        value_loss = 0.5 * torch.mean((mb_returns - values) ** 2)

        has_actors = bool(actor_idx)
        if has_actors:
            mb_actor = [actor_transitions[p] for p in actor_idx]
            new_log_probs, entropies = self.policy.evaluate_transitions(mb_actor)
            mb_old_log_probs = old_log_probs[actor_idx]
            mb_advantages = advantages_t[actor_idx]
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            )
            policy_loss = -torch.min(
                ratio * mb_advantages,
                clipped_ratio * mb_advantages,
            ).mean()
            entropy = entropies.mean()
            normalized_entropy = normalize_entropy(
                entropies=entropies,
                transitions=mb_actor,
            )
        else:
            policy_loss = torch.zeros((), device=self.policy.device)
            entropy = torch.zeros((), device=self.policy.device)

        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.actors.parameters(),
            self.config.max_grad_norm,
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        if not has_actors:
            return None

        with torch.no_grad():
            approx_kl = (mb_old_log_probs - new_log_probs).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                .float()
                .mean()
            )
        return (
            loss.item(),
            policy_loss.item(),
            value_loss.item(),
            entropy.item(),
            approx_kl.item(),
            clip_fraction.item(),
            actor_grad_norm.item(),
            critic_grad_norm.item(),
            normalized_entropy.item(),
            ratio.std(unbiased=False).item(),
            torch.max(torch.abs(ratio - 1.0)).item(),
        )


def compute_joint_advantages(
    *,
    transitions: list[JointMacroTransition],
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    reward_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    returns = np.zeros(len(transitions), dtype=np.float32)

    by_episode: dict[int, list[int]] = {}
    for idx, transition in enumerate(transitions):
        by_episode.setdefault(transition.episode_id, []).append(idx)

    rewards = np.array(
        [transition.reward * reward_scale for transition in transitions],
        dtype=np.float32,
    )
    delta_t = np.array([transition.elapsed_time for transition in transitions], dtype=np.float32)
    dones = np.array(
        [transition.terminated or transition.truncated for transition in transitions],
        dtype=bool,
    )

    for indices in by_episode.values():
        episode_advantages, episode_returns = compute_time_scaled_gae(
            rewards=rewards[indices],
            values=values[indices],
            next_values=next_values[indices],
            delta_t=delta_t[indices],
            dones=dones[indices],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages[indices] = episode_advantages
        returns[indices] = episode_returns

    return advantages, returns


def map_actor_advantages(
    *,
    transitions: list[AgentTransition],
    joint_advantages: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [joint_advantages[transition.joint_index] for transition in transitions],
        dtype=np.float32,
    )


def _transition_actor_hidden(
    transition: AgentTransition,
    *,
    hidden_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if transition.actor_hidden is None:
        return torch.zeros(hidden_dim, dtype=torch.float32, device=device)
    return torch.as_tensor(transition.actor_hidden, dtype=torch.float32, device=device)


def _chunk(items: list[int], num_chunks: int) -> list[list[int]]:
    """Split ``items`` into up to ``num_chunks`` roughly equal contiguous parts.

    Every item appears in exactly one part; empty parts are dropped. With
    ``num_chunks == 1`` this returns ``[items]`` so a full-batch update is
    numerically identical to the non-minibatched path.
    """
    if num_chunks <= 1 or len(items) <= 1:
        return [items]
    size = -(-len(items) // num_chunks)  # ceil division
    return [items[start : start + size] for start in range(0, len(items), size)]


def normalize_advantages(advantages: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def advantage_group_ids(
    actor_transitions: list[AgentTransition],
    rollout: RolloutBuffer,
    mode: str,
) -> list[Any]:
    """Group key per actor transition for grouped advantage normalization.

    ``per_scenario`` keeps all episodes of one workload scenario in a single
    group (preserving the genuine between-episode advantage spread within a
    scenario while equalizing across scenarios of different reward magnitude).
    ``per_episode`` groups each rollout episode on its own. ``per_scenario``
    falls back to per-episode grouping when the rollout carries no scenario
    labels (e.g. a raw single-scenario buffer in tests).
    """
    scenarios = getattr(rollout, "episode_scenarios", None) or {}
    if mode == "per_scenario" and scenarios:
        return [
            scenarios.get(transition.episode_id, str(transition.episode_id))
            for transition in actor_transitions
        ]
    return [transition.episode_id for transition in actor_transitions]


def normalize_advantages_by_group(
    advantages: torch.Tensor,
    group_ids: list[Any],
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Standardize advantages within each group (per scenario or per episode).

    A *global* normalization divides every advantage by one shared std, which is
    dominated by the highest-variance scenario (burst_stress rewards are ~6x
    balanced), shrinking the low-magnitude scenarios' advantages toward zero so
    they barely train. Normalizing each group independently makes every scenario
    contribute an equal-scale gradient regardless of its raw reward magnitude. A
    singleton group collapses to 0 (a single sample carries no relative-advantage
    signal).
    """
    if advantages.numel() <= 1:
        return advantages
    groups = np.asarray(group_ids)
    result = advantages.clone()
    for group in np.unique(groups):
        idx = torch.from_numpy(np.nonzero(groups == group)[0]).to(advantages.device)
        values = advantages.index_select(0, idx)
        result[idx] = (values - values.mean()) / (values.std(unbiased=False) + eps)
    return result


def normalize_entropy(
    *,
    entropies: torch.Tensor,
    transitions: list[AgentTransition],
) -> torch.Tensor:
    valid_counts = torch.tensor(
        [np.count_nonzero(transition.action_mask) for transition in transitions],
        dtype=entropies.dtype,
        device=entropies.device,
    )
    max_entropies = torch.log(valid_counts)
    learnable = max_entropies > 0.0
    if not torch.any(learnable):
        return torch.zeros((), dtype=entropies.dtype, device=entropies.device)
    return torch.mean(entropies[learnable] / max_entropies[learnable])


def _stack_tensor_dicts(
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Concatenate a list of per-row tensor dicts along the batch dimension."""
    return {key: torch.cat([row[key] for row in rows], dim=0) for key in rows[0]}


def _raw_rows_to_tensors(
    batch: AgentBatch,
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Gather the given agent rows into fresh, un-normalized tensors.

    numpy fancy-indexing always copies, so the returned tensors never alias the
    source ``AgentBatch`` — ``normalize_observation_tensors`` can mutate them in
    place safely.
    """
    index = np.asarray(rows, dtype=np.int64)
    return {
        "self_features": torch.as_tensor(batch.self_features[index], dtype=torch.float32, device=device),
        "ready_queue": torch.as_tensor(batch.ready_queue[index], dtype=torch.float32, device=device),
        "ready_mask": torch.as_tensor(batch.ready_mask[index], dtype=torch.float32, device=device),
        "other_cores": torch.as_tensor(batch.other_cores[index], dtype=torch.float32, device=device),
        "other_core_mask": torch.as_tensor(batch.other_core_mask[index], dtype=torch.float32, device=device),
        "system": torch.as_tensor(batch.system[index], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(batch.action_mask[index], dtype=torch.bool, device=device),
    }


def batch_rows_to_tensors(
    batch: AgentBatch,
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return normalize_observation_tensors(_raw_rows_to_tensors(batch, rows, device))


def normalize_observation_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep heterogeneous simulator features in a stable range for MLP inputs.

    Mutates ``tensors`` in place — the builders above hand it freshly allocated
    tensors. Every step is per-row independent, so normalizing a stacked batch
    once is identical to normalizing each row then stacking; the update path
    relies on this to batch many transitions through a single call.
    """

    self_features = tensors["self_features"]
    # self = [core_type, busy, elapsed, energy, dt_since,
    #         running_latency_class, running_cpu_intensity, running_cpu_progress]
    self_features[:, 0] /= max(len(CoreType) - 1, 1)
    self_features[:, 2:5] = torch.log1p(self_features[:, 2:5])
    self_features[:, 5] /= 2.0
    self_features[:, 7] = torch.log1p(self_features[:, 7])

    ready_queue = tensors["ready_queue"]
    # [waiting_time, cpu_progress, latency_class, cpu_intensity]
    ready_queue[:, :, 0:2] = torch.log1p(ready_queue[:, :, 0:2])
    ready_queue[:, :, 2] /= 2.0

    other_cores = tensors["other_cores"]
    if other_cores.shape[1] > 0:
        other_cores[:, :, 0] /= max(len(CoreType) - 1, 1)
        other_cores[:, :, 2] = torch.log1p(other_cores[:, :, 2])

    system = tensors["system"]
    system[:, 0] /= 12.0
    system[:, 2:] /= 12.0
    return tensors


def _actor_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "self_features": tensors["self_features"],
        "ready_queue": tensors["ready_queue"],
        "ready_mask": tensors["ready_mask"],
        "other_cores": tensors["other_cores"],
        "other_core_mask": tensors["other_core_mask"],
        "system": tensors["system"],
        "action_mask": tensors["action_mask"],
    }


def _critic_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "self_features": tensors["self_features"],
        "ready_queue": tensors["ready_queue"],
        "ready_mask": tensors["ready_mask"],
        "other_cores": tensors["other_cores"],
        "other_core_mask": tensors["other_core_mask"],
        "system": tensors["system"],
    }
