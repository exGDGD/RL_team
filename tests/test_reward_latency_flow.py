"""LATENCY_FLOW reward mode: priority-weighted flow time.

These lock the properties that motivated the mode: idle is never free (the
degenerate optimum of the cost-only objective is gone), and a shorter-turnaround
scheduler scores strictly higher.
"""

import numpy as np
import pytest

from src.env import CoreType, RewardWeights, SchedulerEnv, WorkloadScenario
from src.baselines import RandomPolicy, SJFLikePolicy, run_episode
from src.rl import collect_episode


class NoOpPolicy:
    def act(self, batch):
        return (
            {agent_id: 0 for agent_id in batch.agent_ids},
            {agent_id: 0.0 for agent_id in batch.agent_ids},
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


def _make(seed: int, **weight_overrides) -> SchedulerEnv:
    weights = RewardWeights(energy=0.0, flow_time=1.0, **weight_overrides)
    return SchedulerEnv(
        core_config={CoreType.P: 2, CoreType.E: 2},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=1.0,
        episode_time=30.0,
        max_tasks=16,
        seed=seed,
        enable_preemption=False,
        reward_weights=weights,
        reward_mode="latency_flow",
    )


def test_idle_is_heavily_penalized_not_free() -> None:
    """The whole point: a never-dispatching policy must bleed flow-time penalty
    every step instead of scoring ~0 (the old cost-only degenerate optimum)."""

    idle = collect_episode(_make(3), NoOpPolicy(), seed=3)
    working = collect_episode(_make(3), FirstValidPolicy(), seed=3)

    assert idle.total_env_reward < working.total_env_reward
    # Far from the old "idle == 0" optimum.
    assert idle.total_env_reward < -100.0


def test_sjf_outperforms_random_under_flow_reward() -> None:
    """Flow time rewards shorter turnaround, so SJF should beat random."""

    sjf = np.mean(
        [run_episode(_make(s), SJFLikePolicy(), seed=s).total_reward for s in range(100, 105)]
    )
    rnd = np.mean(
        [
            run_episode(_make(s), RandomPolicy(seed=1), seed=s).total_reward
            for s in range(100, 105)
        ]
    )

    assert sjf > rnd


def test_response_weight_scales_the_waiting_penalty() -> None:
    """Under an all-idle trajectory every task is still waiting for its first
    run, so the response-time multiplier scales the whole penalty linearly."""

    base = collect_episode(_make(7, response_weight=1.0), NoOpPolicy(), seed=7)
    tripled = collect_episode(_make(7, response_weight=3.0), NoOpPolicy(), seed=7)

    assert base.total_env_reward < 0.0
    assert tripled.total_env_reward == pytest.approx(3.0 * base.total_env_reward, rel=0.02)
