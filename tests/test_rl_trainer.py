import numpy as np
import pytest
from argparse import Namespace
from pathlib import Path

torch = pytest.importorskip("torch")

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import AgentBatch, collect_episode
from src.rl.trainer import (
    ACACConfig,
    ACACTrainer,
    TorchACACPolicy,
    compute_advantages,
)
from src.train_acac import append_jsonl, save_checkpoint


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


def test_checkpoint_and_jsonl_log_can_be_written(tmp_path: Path) -> None:
    policy = TorchACACPolicy(ACACConfig(hidden_dim=16, critic_heads=4))
    trainer = ACACTrainer(policy)
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "latest.pt"

    append_jsonl(metrics_path, {"episode": 1, "reward": 3.5})
    save_checkpoint(
        torch=torch,
        path=checkpoint_path,
        episode_idx=1,
        policy=policy,
        trainer=trainer,
        config=policy.config,
        args=Namespace(output_dir=tmp_path, resume=None),
        best_eval_reward=3.5,
        eval_summary={"reward": 3.5},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics_path.read_text(encoding="utf-8").endswith("\n")
    assert checkpoint["episode"] == 1
    assert checkpoint["best_eval_reward"] == pytest.approx(3.5)
    assert checkpoint["args"]["output_dir"] == str(tmp_path)
