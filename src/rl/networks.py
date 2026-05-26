from __future__ import annotations

import torch
from torch import nn

from src.env.spaces import (
    OTHER_CORE_FEATURE_DIM,
    READY_TASK_FEATURE_DIM,
    SELF_FEATURE_DIM,
    SYSTEM_FEATURE_DIM,
)


class TypeSharedActor(nn.Module):
    """Actor used by one core type group.

    The same module can be reused for every core of a given type. It scores
    NO-OP separately and scores each ready-queue slot with a shared task head.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.self_encoder = _mlp(SELF_FEATURE_DIM, hidden_dim, hidden_dim)
        self.system_encoder = _mlp(SYSTEM_FEATURE_DIM, hidden_dim, hidden_dim)
        self.other_encoder = _mlp(OTHER_CORE_FEATURE_DIM, hidden_dim, hidden_dim)
        self.task_encoder = _mlp(READY_TASK_FEATURE_DIM, hidden_dim, hidden_dim)
        self.context = _mlp(hidden_dim * 3, hidden_dim, hidden_dim)
        self.noop_head = nn.Linear(hidden_dim, 1)
        self.task_head = _mlp(hidden_dim * 2, hidden_dim, 1, final_activation=False)

    def forward(
        self,
        self_features: torch.Tensor,
        ready_queue: torch.Tensor,
        ready_mask: torch.Tensor,
        other_cores: torch.Tensor,
        other_core_mask: torch.Tensor,
        system: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_emb = self.self_encoder(self_features)
        system_emb = self.system_encoder(system)
        other_emb = masked_mean(
            self.other_encoder(other_cores),
            other_core_mask,
            dim=1,
        )
        context = self.context(torch.cat([self_emb, system_emb, other_emb], dim=-1))

        task_emb = self.task_encoder(ready_queue)
        expanded_context = context.unsqueeze(1).expand(-1, ready_queue.shape[1], -1)
        task_logits = self.task_head(torch.cat([task_emb, expanded_context], dim=-1))
        task_logits = task_logits.squeeze(-1)
        noop_logits = self.noop_head(context)
        logits = torch.cat([noop_logits, task_logits], dim=-1)
        if action_mask is not None:
            logits = mask_logits(logits, action_mask)
        return logits


class AgentCentricCritic(nn.Module):
    """Centralized attention critic that returns one value per agent row."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.self_encoder = _mlp(SELF_FEATURE_DIM, hidden_dim, hidden_dim)
        self.system_encoder = _mlp(SYSTEM_FEATURE_DIM, hidden_dim, hidden_dim)
        self.other_encoder = _mlp(OTHER_CORE_FEATURE_DIM, hidden_dim, hidden_dim)
        self.task_encoder = _mlp(READY_TASK_FEATURE_DIM, hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.value_head = _mlp(hidden_dim * 4, hidden_dim, 1, final_activation=False)

    def forward(
        self,
        self_features: torch.Tensor,
        ready_queue: torch.Tensor,
        ready_mask: torch.Tensor,
        other_cores: torch.Tensor,
        other_core_mask: torch.Tensor,
        system: torch.Tensor,
    ) -> torch.Tensor:
        self_emb = self.self_encoder(self_features)
        system_emb = self.system_encoder(system)
        task_summary = masked_mean(self.task_encoder(ready_queue), ready_mask, dim=1)
        if other_cores.shape[1] == 0:
            attended = torch.zeros_like(self_emb)
        else:
            other_emb = self.other_encoder(other_cores)
            key_padding_mask = ~other_core_mask.bool()
            attended, _ = self.attention(
                query=self_emb.unsqueeze(1),
                key=other_emb,
                value=other_emb,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attended = attended.squeeze(1)
        value_input = torch.cat([self_emb, system_emb, task_summary, attended], dim=-1)
        return self.value_head(value_input).squeeze(-1)


def mask_logits(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    invalid_value: float = -1.0e9,
) -> torch.Tensor:
    return logits.masked_fill(~action_mask.bool(), invalid_value)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype).unsqueeze(-1)
    masked_values = values * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    return masked_values.sum(dim=dim) / denom


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    final_activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
