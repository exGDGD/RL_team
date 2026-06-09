"""Parallel-rollout correctness (the parts that do not require torch).

The torch worker itself can only be exercised on a machine with torch (Colab),
but the risky logic — merging episodes in seed order and shipping buffers across
a process boundary — is pure-Python/numpy and is validated here.
"""

import pickle
from argparse import Namespace

import pytest

from src.env.metrics import EpisodeMetrics
from src.rl import RolloutBuffer, collect_episode
from src.train_acac import (
    aggregate_episode_metrics,
    collect_training_rollout,
    make_env,
    rollout_seeds,
)


class FirstValidPolicy:
    def act(self, batch):
        actions = {}
        log_probs = {}
        for row, agent_id in enumerate(batch.agent_ids):
            valid = [
                idx for idx, is_valid in enumerate(batch.action_mask[row]) if idx > 0 and is_valid
            ]
            actions[agent_id] = valid[0] if bool(batch.decision_mask[row]) and valid else 0
            log_probs[agent_id] = 0.0
        return actions, log_probs


def _args() -> Namespace:
    return Namespace(
        seed=0,
        rollout_episodes=4,
        arrival_rate=1.0,
        episode_time=30.0,
        max_tasks=16,
        enable_preemption=True,
        progress_work=0.0,
        completion=0.0,
        completion_work=0.0,
        lambda_energy=0.1,
        lambda_starvation=0.05,
        lambda_latency=0.5,
        starvation_max_wait_weight=0.5,
    )


def test_sequential_training_rollout_matches_manual_loop() -> None:
    args = _args()
    policy = FirstValidPolicy()

    rollout, metrics = collect_training_rollout(
        policy, args, gamma=0.99, episode_idx=1, executor=None
    )

    manual = RolloutBuffer()
    manual_metrics = []
    for seed in rollout_seeds(args, 1):
        env = make_env(args, seed=seed)
        manual.extend(collect_episode(env, policy, seed=seed, gamma=0.99))
        manual_metrics.append(env.metrics())

    assert rollout.episodes == manual.episodes == args.rollout_episodes
    assert len(rollout) == len(manual)
    assert [t.episode_id for t in rollout.transitions] == [
        t.episode_id for t in manual.transitions
    ]
    assert [t.joint_index for t in rollout.transitions] == [
        t.joint_index for t in manual.transitions
    ]
    assert rollout.total_env_reward == manual.total_env_reward
    # collect_training_rollout now returns the per-field mean across all rollout
    # episodes, not the last-seed snapshot.
    assert metrics == aggregate_episode_metrics(manual_metrics)


def test_seed_order_merge_is_completion_order_independent() -> None:
    """Workers finish out of order; merging in seed order must still reproduce
    the exact episode ids and joint indices of a sequential rollout."""

    args = _args()
    policy = FirstValidPolicy()
    seeds = rollout_seeds(args, 1)

    def collect(seed: int) -> RolloutBuffer:
        return collect_episode(make_env(args, seed=seed), policy, seed=seed, gamma=0.99)

    forward = RolloutBuffer()
    for seed in seeds:
        forward.extend(collect(seed))

    # Simulate workers completing in reverse order, then merge in seed order.
    out_of_order = {seed: collect(seed) for seed in reversed(seeds)}
    reordered = RolloutBuffer()
    for seed in seeds:
        reordered.extend(out_of_order[seed])

    assert [t.episode_id for t in forward.transitions] == [
        t.episode_id for t in reordered.transitions
    ]
    assert [t.joint_index for t in forward.transitions] == [
        t.joint_index for t in reordered.transitions
    ]
    assert len(forward.joint_transitions) == len(reordered.joint_transitions)
    assert forward.total_env_reward == reordered.total_env_reward


def _episode_metrics(
    *,
    total: int,
    completed: int,
    throughput: float,
    turnaround: float | None,
    p99_turnaround: float | None,
    per_core: dict[str, float],
) -> EpisodeMetrics:
    return EpisodeMetrics(
        total_tasks=total,
        completed_tasks=completed,
        makespan=40.0,
        throughput=throughput,
        total_energy=100.0,
        mean_response_time=1.0,
        p95_response_time=2.0,
        p99_response_time=3.0,
        mean_turnaround_time=turnaround,
        p95_turnaround_time=turnaround,
        p99_turnaround_time=p99_turnaround,
        mean_ready_wait_time=1.0,
        p95_ready_wait_time=2.0,
        starvation_rate=None,
        mean_utilization=sum(per_core.values()) / len(per_core),
        per_core_utilization=per_core,
    )


def test_aggregate_episode_metrics_averages_fields_and_handles_none() -> None:
    metrics = [
        _episode_metrics(
            total=32, completed=30, throughput=0.5, turnaround=10.0,
            p99_turnaround=None, per_core={"p_0": 0.8, "e_0": 0.6},
        ),
        _episode_metrics(
            total=32, completed=32, throughput=0.7, turnaround=20.0,
            p99_turnaround=5.0, per_core={"p_0": 0.6, "e_0": 0.4},
        ),
    ]

    agg = aggregate_episode_metrics(metrics)

    # Counts become floats; plain numeric fields are averaged.
    assert agg["completed_tasks"] == pytest.approx(31.0)
    assert agg["total_tasks"] == pytest.approx(32.0)
    assert agg["throughput"] == pytest.approx(0.6)
    assert agg["mean_turnaround_time"] == pytest.approx(15.0)
    # None entries are dropped from the mean rather than poisoning it.
    assert agg["p99_turnaround_time"] == pytest.approx(5.0)
    # A field that is None in every episode stays None.
    assert agg["starvation_rate"] is None
    # Nested per-core utilization is averaged per core.
    assert agg["per_core_utilization"] == {
        "p_0": pytest.approx(0.7),
        "e_0": pytest.approx(0.5),
    }


def test_aggregate_episode_metrics_empty_returns_empty_dict() -> None:
    assert aggregate_episode_metrics([]) == {}


def test_rollout_buffer_and_metrics_survive_pickle() -> None:
    """Workers return (buffer, metrics) across a process boundary, so both must
    pickle cleanly with no torch tensors or lambdas inside."""

    args = _args()
    env = make_env(args, seed=7)
    buffer = collect_episode(env, FirstValidPolicy(), seed=7, gamma=0.99)
    metrics = env.metrics()

    restored_buffer = pickle.loads(pickle.dumps(buffer))
    restored_metrics = pickle.loads(pickle.dumps(metrics))

    assert len(restored_buffer) == len(buffer)
    assert restored_buffer.env_steps == buffer.env_steps
    assert [t.episode_id for t in restored_buffer.transitions] == [
        t.episode_id for t in buffer.transitions
    ]
    assert restored_metrics.completed_tasks == metrics.completed_tasks
