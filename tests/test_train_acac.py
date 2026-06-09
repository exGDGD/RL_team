from argparse import Namespace

import pytest

from src.train_acac import (
    CHECKPOINT_VERSION,
    entropy_coef_at,
    reward_weights_from_args,
    validate_checkpoint_version,
)


def test_training_reward_defaults_to_cost_only_objective() -> None:
    weights = reward_weights_from_args(Namespace())

    assert weights.progress_work == 0.0
    assert weights.completion == 0.0
    assert weights.completion_work == 0.0
    assert weights.energy == 0.1
    assert weights.starvation == 0.05
    assert weights.latency == 0.5


def test_training_reward_shaping_can_be_enabled_explicitly() -> None:
    weights = reward_weights_from_args(
        Namespace(
            progress_work=1.0,
            completion=5.0,
            completion_work=5.0,
        )
    )

    assert weights.progress_work == 1.0
    assert weights.completion == 5.0
    assert weights.completion_work == 5.0


def test_resume_rejects_checkpoint_from_another_training_algorithm() -> None:
    with pytest.raises(SystemExit, match="Checkpoint version mismatch"):
        validate_checkpoint_version({})


def test_resume_accepts_current_checkpoint_version() -> None:
    validate_checkpoint_version({"checkpoint_version": CHECKPOINT_VERSION})


def test_entropy_coef_constant_when_no_final_given() -> None:
    args = Namespace(entropy_coef=0.01, entropy_coef_final=None, episodes=100)

    assert entropy_coef_at(args, 1) == pytest.approx(0.01)
    assert entropy_coef_at(args, 50) == pytest.approx(0.01)
    assert entropy_coef_at(args, 100) == pytest.approx(0.01)


def test_entropy_coef_anneals_linearly_to_final() -> None:
    args = Namespace(
        entropy_coef=0.02,
        entropy_coef_final=0.0,
        entropy_anneal_episodes=None,
        episodes=11,
    )

    assert entropy_coef_at(args, 1) == pytest.approx(0.02)
    assert entropy_coef_at(args, 6) == pytest.approx(0.01)
    assert entropy_coef_at(args, 11) == pytest.approx(0.0)


def test_entropy_coef_holds_final_value_after_anneal_window() -> None:
    args = Namespace(
        entropy_coef=0.02,
        entropy_coef_final=0.005,
        entropy_anneal_episodes=10,
        episodes=100,
    )

    assert entropy_coef_at(args, 10) == pytest.approx(0.005)
    # Past the anneal window the coefficient is clamped to the final value.
    assert entropy_coef_at(args, 80) == pytest.approx(0.005)
