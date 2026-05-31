from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.env import CoreType

from .buffer import AgentTransition, RolloutBuffer, compute_time_scaled_gae
from .networks import AgentCentricCritic, TypeSharedActor
from .obs import AgentBatch


@dataclass(frozen=True)
class ACACConfig:
    hidden_dim: int = 128
    critic_heads: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3.0e-4
    allow_noop: bool = False
    update_epochs: int = 4


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


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

    def act(
        self,
        batch: AgentBatch,
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, np.ndarray]]:
        actions = {agent_id: 0 for agent_id in batch.agent_ids}
        log_probs = {agent_id: 0.0 for agent_id in batch.agent_ids}
        effective_masks: dict[str, np.ndarray] = {}
        claimed_slots: set[int] = set()

        with torch.no_grad():
            for row, agent_id in enumerate(batch.agent_ids):
                if not bool(batch.decision_mask[row]):
                    continue

                core_type = list(CoreType)[int(batch.core_type_indices[row])]
                tensors = batch_rows_to_tensors(batch, [row], self.device)
                tensors = self._apply_policy_action_mask(tensors)
                for claimed_slot in claimed_slots:
                    tensors["action_mask"][:, claimed_slot] = False
                if not torch.any(tensors["action_mask"]):
                    continue
                logits = self.actors[core_type.value](**tensors)
                dist = Categorical(logits=logits)
                sampled_actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                sampled_log_probs = dist.log_prob(sampled_actions)

                action = int(sampled_actions[0].item())
                actions[agent_id] = action
                log_probs[agent_id] = float(sampled_log_probs[0].item())
                effective_masks[agent_id] = tensors["action_mask"][0].cpu().numpy()
                if action > 0:
                    claimed_slots.add(action)

        return actions, log_probs, effective_masks

    def evaluate_transitions(
        self,
        transitions: list[AgentTransition],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for idx, transition in enumerate(transitions):
            tensors = transition_row_to_tensors(transition, self.device)
            tensors["action_mask"] = torch.as_tensor(
                transition.action_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            core_type = list(CoreType)[int(transition.obs.core_type_indices[transition.agent_index])]
            logits = self.actors[core_type.value](**_actor_inputs(tensors))
            dist = Categorical(logits=logits)
            action = torch.tensor([transition.action], dtype=torch.long, device=self.device)
            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            values.append(self.critic(**_critic_inputs(tensors)).squeeze(0))

        if not transitions:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)

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

    def values_for_transitions(
        self,
        transitions: list[AgentTransition],
        *,
        next_obs: bool = False,
    ) -> torch.Tensor:
        values = []
        with torch.no_grad():
            for transition in transitions:
                tensors = transition_row_to_tensors(
                    transition,
                    self.device,
                    next_obs=next_obs,
                )
                values.append(self.critic(**_critic_inputs(tensors)).squeeze(0))
        if not values:
            return torch.empty(0, device=self.device)
        return torch.stack(values)


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
            policy.parameters(),
            lr=self.config.learning_rate,
        )

    def update(self, rollout: RolloutBuffer) -> UpdateStats:
        transitions = rollout.transitions
        if not transitions:
            raise ValueError("Cannot update from an empty rollout buffer.")

        with torch.no_grad():
            old_values = self.policy.values_for_transitions(transitions)
            next_values = self.policy.values_for_transitions(transitions, next_obs=True)
            advantages, returns = compute_advantages(
                transitions=transitions,
                values=old_values.cpu().numpy(),
                next_values=next_values.cpu().numpy(),
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )

        old_log_probs = torch.tensor(
            [transition.log_prob for transition in transitions],
            dtype=torch.float32,
            device=self.policy.device,
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.policy.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)
        advantages_t = normalize_advantages(advantages_t)

        epoch_stats = []
        for _ in range(self.config.update_epochs):
            new_log_probs, entropies, values = self.policy.evaluate_transitions(transitions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            )
            policy_loss = -torch.min(
                ratio * advantages_t,
                clipped_ratio * advantages_t,
            ).mean()
            value_loss = 0.5 * torch.mean((returns_t - values) ** 2)
            entropy = entropies.mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                    .float()
                    .mean()
                )
            epoch_stats.append(
                (
                    loss.item(),
                    policy_loss.item(),
                    value_loss.item(),
                    entropy.item(),
                    approx_kl.item(),
                    clip_fraction.item(),
                )
            )

        means = np.mean(epoch_stats, axis=0)
        return UpdateStats(
            loss=float(means[0]),
            policy_loss=float(means[1]),
            value_loss=float(means[2]),
            entropy=float(means[3]),
            approx_kl=float(means[4]),
            clip_fraction=float(means[5]),
        )


def compute_advantages(
    *,
    transitions: list[AgentTransition],
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    returns = np.zeros(len(transitions), dtype=np.float32)

    by_agent: dict[str, list[int]] = {}
    for idx, transition in enumerate(transitions):
        by_agent.setdefault(transition.agent_id, []).append(idx)

    rewards = np.array([transition.reward for transition in transitions], dtype=np.float32)
    delta_t = np.array([transition.elapsed_time for transition in transitions], dtype=np.float32)
    dones = np.array(
        [transition.terminated or transition.truncated for transition in transitions],
        dtype=bool,
    )

    for indices in by_agent.values():
        agent_advantages, agent_returns = compute_time_scaled_gae(
            rewards=rewards[indices],
            values=values[indices],
            next_values=next_values[indices],
            delta_t=delta_t[indices],
            dones=dones[indices],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages[indices] = agent_advantages
        returns[indices] = agent_returns

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def batch_rows_to_tensors(
    batch: AgentBatch,
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
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


def transition_row_to_tensors(
    transition: AgentTransition,
    device: torch.device,
    *,
    next_obs: bool = False,
) -> dict[str, torch.Tensor]:
    batch = transition.next_obs if next_obs else transition.obs
    row = transition.next_agent_index if next_obs else transition.agent_index
    return batch_rows_to_tensors(batch, [row], device)


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
