from argparse import Namespace

import pytest

from src.train_acac import (
    CHECKPOINT_VERSION,
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
