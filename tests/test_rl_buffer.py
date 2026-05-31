import numpy as np
import pytest

from src.rl import JointMacroTransition, RolloutBuffer, compute_time_scaled_gae


def test_time_scaled_gae_matches_one_step_td_when_lambda_zero() -> None:
    advantages, returns = compute_time_scaled_gae(
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        values=np.array([0.5, 0.25], dtype=np.float32),
        next_values=np.array([0.25, 0.0], dtype=np.float32),
        delta_t=np.array([2.0, 1.0], dtype=np.float32),
        dones=np.array([False, True]),
        gamma=0.9,
        gae_lambda=0.0,
    )

    assert advantages[0] == pytest.approx(1.0 + 0.9**2 * 0.25 - 0.5)
    assert advantages[1] == pytest.approx(2.0 - 0.25)
    assert returns[0] == pytest.approx(advantages[0] + 0.5)
    assert returns[1] == pytest.approx(advantages[1] + 0.25)


def test_time_scaled_gae_recurses_with_time_discount() -> None:
    advantages, _ = compute_time_scaled_gae(
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        next_values=np.array([0.0, 0.0], dtype=np.float32),
        delta_t=np.array([2.0, 1.0], dtype=np.float32),
        dones=np.array([False, True]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert advantages[1] == pytest.approx(1.0)
    assert advantages[0] == pytest.approx(1.0 + 0.5**2 * 1.0)


def test_time_scaled_gae_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        compute_time_scaled_gae(
            rewards=np.array([1.0, 2.0]),
            values=np.array([0.0]),
            next_values=np.array([0.0, 0.0]),
            delta_t=np.array([1.0, 1.0]),
            dones=np.array([False, False]),
        )


def test_rollout_buffer_extend_aggregates_diagnostics() -> None:
    target = RolloutBuffer(
        episodes=1,
        total_env_reward=2.5,
        env_steps=2,
        conflicts=1,
        invalid_actions=1,
        decisions=3,
        forced_decisions=2,
        total_task_choices=5,
        max_task_choices=2,
    )
    other = RolloutBuffer(
        episodes=2,
        total_env_reward=3.5,
        env_steps=4,
        conflicts=2,
        invalid_actions=0,
        decisions=2,
        forced_decisions=1,
        total_task_choices=7,
        max_task_choices=5,
    )

    target.extend(other)

    assert target.episodes == 3
    assert target.total_env_reward == pytest.approx(6.0)
    assert target.env_steps == 6
    assert target.conflicts == 3
    assert target.invalid_actions == 1
    assert target.decisions == 5
    assert target.forced_decisions == 3
    assert target.mean_task_choices == pytest.approx(12 / 5)
    assert target.max_task_choices == 5


def test_rollout_buffer_extend_offsets_joint_episode_ids() -> None:
    target = RolloutBuffer(
        episodes=1,
        joint_transitions=[
            JointMacroTransition(
                episode_id=0,
                obs=object(),
                reward=1.0,
                next_obs=object(),
                elapsed_time=1.0,
                terminated=True,
                truncated=False,
            )
        ],
    )
    other = RolloutBuffer(
        episodes=1,
        joint_transitions=[
            JointMacroTransition(
                episode_id=0,
                obs=object(),
                reward=2.0,
                next_obs=object(),
                elapsed_time=1.0,
                terminated=True,
                truncated=False,
            )
        ],
    )

    target.extend(other)

    assert [transition.episode_id for transition in target.joint_transitions] == [0, 1]
