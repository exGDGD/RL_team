from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env import SchedulerEnv

from .obs import AgentBatch, build_agent_batch


@dataclass(frozen=True)
class ImitationExample:
    """One actor decision labeled by an expert scheduling policy."""

    obs: AgentBatch
    agent_index: int
    action_mask: np.ndarray
    action: int


def collect_sjf_examples(
    env: SchedulerEnv,
    *,
    seed: int | None = None,
    only_learnable: bool = True,
    max_env_steps: int = 10_000,
) -> list[ImitationExample]:
    """Collect actor rows labeled by the SJF-like baseline.

    Labels follow the same deterministic core order and claimed-slot masking
    used by the actor rollout path.
    """

    observations, _ = env.reset(seed=seed)
    examples: list[ImitationExample] = []

    for _ in range(max_env_steps):
        batch = build_agent_batch(observations, agent_order=env.agents)
        actions = {agent_id: 0 for agent_id in env.agents}
        available = list(range(1, min(env.queue_size, len(env.ready_queue)) + 1))

        for agent_index, agent_id in enumerate(batch.agent_ids):
            core = env.cores[agent_id]
            if core.busy or not available:
                continue

            action_mask = batch.action_mask[agent_index].astype(bool).copy()
            action_mask[0] = False
            unavailable = set(range(1, env.queue_size + 1)) - set(available)
            for action in unavailable:
                action_mask[action] = False

            valid_actions = [action for action in available if action_mask[action]]
            if not valid_actions:
                continue
            action = min(
                valid_actions,
                key=lambda candidate: env._runtime_on_core(
                    core,
                    env.ready_queue[candidate - 1],
                ),
            )
            if not only_learnable or len(valid_actions) > 1:
                examples.append(
                    ImitationExample(
                        obs=batch,
                        agent_index=agent_index,
                        action_mask=action_mask,
                        action=action,
                    )
                )
            actions[agent_id] = action
            available.remove(action)

        observations, _, terminations, truncations, _ = env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            return examples

    return examples
