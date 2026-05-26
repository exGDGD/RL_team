import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import AgentBatch, collect_episode
from src.rl.trainer import (
    ACACConfig,
    ACACTrainer,
    TorchACACPolicy,
    compute_advantages,
)


class FirstValidPolicy:
    def act(self, batch: AgentBatch) -> tuple[dict[str, int], dict[str, float]]:
        actions = {}
        log_probs = {}
        for row, agent_id in enumerate(batch.agent_ids):
            valid = [idx for idx, is_valid in enumerate(batch.action_mask[row]) if idx > 0 and is_valid]
            actions[agent_id] = valid[0] if bool(batch.decision_mask[row]) and valid else 0
            log_probs[agent_id] = 0.0
        return actions, log_probs


def test_compute_advantages_groups_transitions_by_agent() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )
    rollout = collect_episode(env, FirstValidPolicy(), seed=3)
    values = np.zeros(len(rollout.transitions), dtype=np.float32)
    next_values = np.zeros(len(rollout.transitions), dtype=np.float32)

    advantages, returns = compute_advantages(
        transitions=rollout.transitions,
        values=values,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (len(rollout.transitions),)
    assert returns.shape == (len(rollout.transitions),)


def test_torch_acac_policy_can_update_from_collected_rollout() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )
    rollout = collect_episode(env, FirstValidPolicy(), seed=3)
    policy = TorchACACPolicy(ACACConfig(hidden_dim=16, critic_heads=4))
    trainer = ACACTrainer(policy)

    stats = trainer.update(rollout)

    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
