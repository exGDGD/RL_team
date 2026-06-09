from argparse import Namespace

import pytest

from src.train_acac import (
    CHECKPOINT_VERSION,
    count_labels,
    parse_workload_scenarios,
    reward_weights_from_args,
    rollout_scenarios,
    serialize_args,
    training_scenario_counts,
    validate_checkpoint_version,
)
from src.env import WorkloadScenario


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


def test_parse_workload_scenarios_supports_all_and_lists() -> None:
    assert parse_workload_scenarios("balanced,ui_heavy") == (
        WorkloadScenario.BALANCED,
        WorkloadScenario.UI_HEAVY,
    )
    assert parse_workload_scenarios("all") == tuple(WorkloadScenario)


def test_training_rollout_scenarios_round_robin_across_updates() -> None:
    args = Namespace(
        rollout_episodes=5,
        train_scenarios=(
            WorkloadScenario.BALANCED,
            WorkloadScenario.UI_HEAVY,
            WorkloadScenario.BG_HEAVY,
        ),
    )

    assert rollout_scenarios(args, episode_idx=1) == [
        WorkloadScenario.BALANCED,
        WorkloadScenario.UI_HEAVY,
        WorkloadScenario.BG_HEAVY,
        WorkloadScenario.BALANCED,
        WorkloadScenario.UI_HEAVY,
    ]
    assert rollout_scenarios(args, episode_idx=2) == [
        WorkloadScenario.BG_HEAVY,
        WorkloadScenario.BALANCED,
        WorkloadScenario.UI_HEAVY,
        WorkloadScenario.BG_HEAVY,
        WorkloadScenario.BALANCED,
    ]
    assert training_scenario_counts(args, episode_idx=1) == {
        "balanced": 2,
        "ui_heavy": 2,
        "bg_heavy": 1,
    }


def test_serialize_args_writes_scenario_values() -> None:
    serialized = serialize_args(
        Namespace(
            train_scenarios=(
                WorkloadScenario.BALANCED,
                WorkloadScenario.BURST_STRESS,
            )
        )
    )

    assert serialized["train_scenarios"] == ["balanced", "burst_stress"]


def test_count_labels_records_eval_scenario_mix() -> None:
    assert count_labels(["balanced", "ui_heavy", "balanced"]) == {
        "balanced": 2,
        "ui_heavy": 1,
    }
