import numpy as np
import pytest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

torch = pytest.importorskip("torch")

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import AgentBatch, ReplayBuffer, build_agent_batch, collect_episode
from src.rl.trainer import (
    ACACConfig,
    ACACTrainer,
    TorchACACPolicy,
    _raw_rows_to_tensors,
    _stack_tensor_dicts,
    advantage_group_ids,
    batch_rows_to_tensors,
    compute_joint_advantages,
    map_actor_advantages,
    normalize_advantages,
    normalize_advantages_by_group,
    normalize_observation_tensors,
)
from src.train_acac import CHECKPOINT_VERSION, append_jsonl, save_checkpoint


class FirstValidPolicy:
    def act(self, batch: AgentBatch) -> tuple[dict[str, int], dict[str, float]]:
        actions = {}
        log_probs = {}
        for row, agent_id in enumerate(batch.agent_ids):
            valid = [idx for idx, is_valid in enumerate(batch.action_mask[row]) if idx > 0 and is_valid]
            actions[agent_id] = valid[0] if bool(batch.decision_mask[row]) and valid else 0
            log_probs[agent_id] = 0.0
        return actions, log_probs


def _sample_batch(seed: int = 5):
    env = SchedulerEnv(
        core_config={CoreType.P: 2, CoreType.E: 2},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=1.0,
        episode_time=30.0,
        max_tasks=16,
        seed=seed,
    )
    observations, _ = env.reset(seed=seed)
    return build_agent_batch(observations, agent_order=env.agents)


def test_normalize_batched_equals_per_row() -> None:
    """The update path normalizes a stacked batch once instead of per row; this
    must be bit-identical to normalizing each row and then stacking."""
    batch = _sample_batch()
    device = torch.device("cpu")
    n = batch.num_agents

    per_row = _stack_tensor_dicts(
        [
            normalize_observation_tensors(_raw_rows_to_tensors(batch, [i], device))
            for i in range(n)
        ]
    )
    batched = normalize_observation_tensors(
        _stack_tensor_dicts([_raw_rows_to_tensors(batch, [i], device) for i in range(n)])
    )

    for key in per_row:
        assert torch.equal(per_row[key], batched[key]), key


def test_batch_rows_to_tensors_does_not_mutate_source_batch() -> None:
    """In-place normalization must not corrupt the stored AgentBatch (numpy
    fancy-indexing copies, so the tensors never alias the source)."""
    batch = _sample_batch()
    keys = ["self_features", "ready_queue", "other_cores", "system"]
    snapshot = {k: np.array(getattr(batch, k), copy=True) for k in keys}

    batch_rows_to_tensors(batch, list(range(batch.num_agents)), torch.device("cpu"))

    for key in keys:
        assert np.array_equal(getattr(batch, key), snapshot[key]), key


def test_compute_advantages_groups_transitions_by_episode() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )
    rollout = collect_episode(env, FirstValidPolicy(), seed=3)
    values = np.zeros(len(rollout.joint_transitions), dtype=np.float32)
    next_values = np.zeros(len(rollout.joint_transitions), dtype=np.float32)

    advantages, returns = compute_joint_advantages(
        transitions=rollout.joint_transitions,
        values=values,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (len(rollout.joint_transitions),)
    assert returns.shape == (len(rollout.joint_transitions),)


def test_torch_acac_policy_can_update_from_collected_rollout() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=2.0,
        episode_time=30.0,
        max_tasks=16,
        seed=3,
    )
    rollout = collect_episode(env, FirstValidPolicy(), seed=3)
    policy = TorchACACPolicy(ACACConfig(hidden_dim=16, critic_heads=4))
    trainer = ACACTrainer(policy)

    stats = trainer.update(rollout)

    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    assert stats.actor_samples > 0
    assert stats.actor_samples == sum(
        np.count_nonzero(transition.action_mask) > 1
        for transition in rollout.transitions
    )


def test_torch_policy_rollout_stores_actor_hidden_state() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=2.0,
        episode_time=30.0,
        max_tasks=8,
        seed=3,
    )
    policy = TorchACACPolicy(ACACConfig(hidden_dim=16, critic_heads=4))

    rollout = collect_episode(env, policy, seed=3)

    assert rollout.transitions
    assert all(transition.actor_hidden is not None for transition in rollout.transitions)
    assert all(transition.actor_hidden.shape == (16,) for transition in rollout.transitions)


def _balanced_rollout(seed: int = 3):
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=2.0,
        episode_time=30.0,
        max_tasks=16,
        seed=seed,
    )
    return collect_episode(env, FirstValidPolicy(), seed=seed)


def _learnable_actor_samples(rollout) -> int:
    return sum(
        np.count_nonzero(transition.action_mask) > 1
        for transition in rollout.transitions
    )


def test_minibatched_update_covers_all_actor_samples() -> None:
    rollout = _balanced_rollout()
    policy = TorchACACPolicy(
        ACACConfig(hidden_dim=16, critic_heads=4, num_minibatches=4)
    )
    trainer = ACACTrainer(policy)

    stats = trainer.update(rollout)

    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    # Splitting into minibatches must still credit every learnable transition.
    assert stats.actor_samples == _learnable_actor_samples(rollout)


def test_update_reports_supplied_entropy_coefficient() -> None:
    rollout = _balanced_rollout()
    policy = TorchACACPolicy(ACACConfig(hidden_dim=16, critic_heads=4))
    trainer = ACACTrainer(policy)

    stats = trainer.update(rollout, entropy_coef=0.05)

    assert stats.entropy_coef == pytest.approx(0.05)


def test_normalize_advantages_by_group_standardizes_each_group() -> None:
    """Each group is centered to zero mean and unit std independently, so two
    groups whose raw magnitudes differ 100x both end at the same scale."""
    advantages = torch.tensor([10.0, 20.0, 30.0, 0.1, 0.2, 0.3])
    groups = ["big", "big", "big", "small", "small", "small"]

    out = normalize_advantages_by_group(advantages, groups)

    for name in ("big", "small"):
        sel = out[[i for i, g in enumerate(groups) if g == name]]
        assert sel.mean().abs().item() < 1e-5
        assert abs(sel.std(unbiased=False).item() - 1.0) < 1e-5


def test_normalize_advantages_by_group_matches_global_for_single_group() -> None:
    """With one group, grouped normalization is identical to the global one."""
    advantages = torch.tensor([1.0, -2.0, 3.0, 0.5])

    grouped = normalize_advantages_by_group(advantages, ["s", "s", "s", "s"])
    glob = normalize_advantages(advantages.clone())

    assert torch.allclose(grouped, glob, atol=1e-6)


def test_advantage_group_ids_prefers_scenarios_then_falls_back_to_episode() -> None:
    rollout = _balanced_rollout(seed=3)
    actor = [t for t in rollout.transitions if np.count_nonzero(t.action_mask) > 1]

    # No scenario labels -> group by episode id.
    assert advantage_group_ids(actor, rollout, "per_scenario") == [
        t.episode_id for t in actor
    ]
    # With labels -> group by scenario value.
    rollout.episode_scenarios = {t.episode_id: "balanced" for t in actor}
    assert advantage_group_ids(actor, rollout, "per_scenario") == ["balanced"] * len(actor)


def test_per_scenario_advantage_norm_runs_update() -> None:
    current = _balanced_rollout(seed=3)
    replay = ReplayBuffer(capacity=1)
    replay.add(_balanced_rollout(seed=4))
    merged = replay.combined(current)
    # Two episodes (ids 0 and 1) tagged as different scenarios.
    merged.episode_scenarios = {0: "balanced", 1: "burst_stress"}

    policy = TorchACACPolicy(
        ACACConfig(hidden_dim=16, critic_heads=4, advantage_norm="per_scenario")
    )
    trainer = ACACTrainer(policy)

    stats = trainer.update(merged)

    assert np.isfinite(stats.loss)
    assert stats.actor_samples == _learnable_actor_samples(merged)


def test_update_on_replayed_rollout_trains_on_merged_samples() -> None:
    current = _balanced_rollout(seed=3)
    replay = ReplayBuffer(capacity=1)
    replay.add(_balanced_rollout(seed=4))
    merged = replay.combined(current)

    policy = TorchACACPolicy(
        ACACConfig(hidden_dim=16, critic_heads=4, num_minibatches=2)
    )
    trainer = ACACTrainer(policy)

    stats = trainer.update(merged)

    assert np.isfinite(stats.loss)
    assert stats.actor_samples == _learnable_actor_samples(merged)
    assert stats.actor_samples > _learnable_actor_samples(current)


def test_compute_advantages_does_not_cross_episode_boundaries() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=2,
        seed=3,
    )
    transition = collect_episode(env, FirstValidPolicy(), seed=3).joint_transitions[0]
    transitions = [
        replace(transition, episode_id=0, reward=1.0, terminated=False),
        replace(transition, episode_id=1, reward=2.0, terminated=False),
    ]

    advantages, _ = compute_joint_advantages(
        transitions=transitions,
        values=np.zeros(2, dtype=np.float32),
        next_values=np.zeros(2, dtype=np.float32),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert advantages.tolist() == pytest.approx([1.0, 2.0])


def test_actor_advantages_use_macro_action_start_joint_index() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=2,
        seed=3,
    )
    transition = collect_episode(env, FirstValidPolicy(), seed=3).transitions[0]
    transitions = [
        replace(transition, joint_index=1),
        replace(transition, joint_index=0),
    ]

    advantages = map_actor_advantages(
        transitions=transitions,
        joint_advantages=np.array([3.0, 7.0], dtype=np.float32),
    )

    assert advantages.tolist() == pytest.approx([7.0, 3.0])


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
    assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
    assert checkpoint["best_eval_reward"] == pytest.approx(3.5)
    assert checkpoint["args"]["output_dir"] == str(tmp_path)
