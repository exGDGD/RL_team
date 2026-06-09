from argparse import Namespace

import pytest

from src.train_acac import (
    CHECKPOINT_VERSION,
    count_labels,
    parse_workload_scenarios,
    entropy_coef_at,
    reward_weights_from_args,
    rollout_scenarios,
    serialize_args,
    summarize_eval_rows,
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


def test_summarize_eval_rows_groups_by_scenario() -> None:
    assert summarize_eval_rows(
        [
            {"scenario": "balanced", "reward": -10.0, "turnaround": 5.0},
            {"scenario": "balanced", "reward": -14.0, "turnaround": None},
            {"scenario": "ui_heavy", "reward": -3.0, "turnaround": 1.0},
        ]
    ) == {
        "balanced": {"reward": -12.0, "turnaround": 5.0},
        "ui_heavy": {"reward": -3.0, "turnaround": 1.0},
    }
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
