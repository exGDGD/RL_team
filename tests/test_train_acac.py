from argparse import Namespace

import pytest

from src.train_acac import (
    CHECKPOINT_VERSION,
    checkpoint_score,
    count_labels,
    parse_workload_scenarios,
    entropy_coef_at,
    lr_scale_at,
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
    # Now also reports per-scenario reward spread (sample std, ddof=1) and count
    # so consumers can form the standard error (std/sqrt(n)).
    assert summarize_eval_rows(
        [
            {"scenario": "balanced", "reward": -10.0, "turnaround": 5.0},
            {"scenario": "balanced", "reward": -14.0, "turnaround": None},
            {"scenario": "ui_heavy", "reward": -3.0, "turnaround": 1.0},
        ]
    ) == {
        "balanced": {
            "reward": -12.0,
            "turnaround": 5.0,
            "n": 2,
            "reward_std": pytest.approx(8.0**0.5),
        },
        "ui_heavy": {"reward": -3.0, "turnaround": 1.0, "n": 1, "reward_std": 0.0},
    }


def _eval_summary_with_scenarios(rl_by_scenario: dict[str, float]) -> dict:
    # Realistic baselines: random, mlfq (mlfq is the best fair one). sjf_like is
    # the clairvoyant oracle and must be EXCLUDED from the score.
    baselines = {
        "random": {"by_scenario": {"a": {"reward": -120.0}, "b": {"reward": -1200.0}}},
        "mlfq": {"by_scenario": {"a": {"reward": -100.0}, "b": {"reward": -1000.0}}},
        "sjf_like": {"by_scenario": {"a": {"reward": -50.0}, "b": {"reward": -500.0}}},
    }
    return {
        "reward": -1.0,
        "by_scenario": {s: {"reward": r} for s, r in rl_by_scenario.items()},
        "baselines": baselines,
    }


def test_checkpoint_score_zero_when_matching_best_realistic_baseline() -> None:
    # Matching mlfq (best realistic) -> 0. If the oracle sjf were the reference
    # this would be strongly negative, so this also proves sjf is excluded.
    summary = _eval_summary_with_scenarios({"a": -100.0, "b": -1000.0})
    assert checkpoint_score(summary) == pytest.approx(0.0)


def test_checkpoint_score_positive_when_beating_best_realistic_baseline() -> None:
    # a: (-90 - -100)/100 = +0.1 ; b: (-950 - -1000)/1000 = +0.05 -> mean 0.075
    summary = _eval_summary_with_scenarios({"a": -90.0, "b": -950.0})
    assert checkpoint_score(summary) == pytest.approx(0.075)


def test_checkpoint_score_weights_fixed_delta_more_in_low_magnitude_scenario() -> None:
    # Same absolute 10-reward miss vs best realistic in both, but it is 0.1 of
    # |best| in the small scenario 'a' and only 0.01 in the big 'b'.
    summary = _eval_summary_with_scenarios({"a": -110.0, "b": -1010.0})
    assert checkpoint_score(summary) == pytest.approx((-0.1 + -0.01) / 2)


def test_checkpoint_score_clamps_a_catastrophic_scenario() -> None:
    # 'a' is 10x worse than best realistic (rel -10) but clamps to -1; 'b' matches
    # best realistic (0) -> mean -0.5 (unclamped would be -5).
    summary = _eval_summary_with_scenarios({"a": -1100.0, "b": -1000.0})
    assert checkpoint_score(summary) == pytest.approx(-0.5)


def test_checkpoint_score_stable_when_baselines_bunch() -> None:
    # Baselines within a few points: dividing by |best realistic| stays bounded.
    summary = {
        "reward": -1.0,
        "by_scenario": {"ui": {"reward": -756.0}},
        "baselines": {
            "random": {"by_scenario": {"ui": {"reward": -711.0}}},
            "mlfq": {"by_scenario": {"ui": {"reward": -705.0}}},
            "sjf_like": {"by_scenario": {"ui": {"reward": -703.0}}},  # oracle, excluded
        },
    }
    # best realistic = mlfq -705, rel = (-756 - -705)/705 = -0.0723
    assert checkpoint_score(summary) == pytest.approx(-51.0 / 705.0)


def test_checkpoint_score_none_without_scenario_baselines() -> None:
    assert checkpoint_score({"reward": -1.0}) is None
    assert (
        checkpoint_score({"reward": -1.0, "by_scenario": {"a": {"reward": -5.0}}})
        is None
    )


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


def test_lr_scale_constant_when_anneal_disabled() -> None:
    args = Namespace(lr_anneal_final_frac=1.0, lr_anneal_episodes=None, episodes=100)

    assert lr_scale_at(args, 1) == pytest.approx(1.0)
    assert lr_scale_at(args, 50) == pytest.approx(1.0)
    assert lr_scale_at(args, 100) == pytest.approx(1.0)


def test_lr_scale_anneals_linearly_to_final_fraction() -> None:
    args = Namespace(lr_anneal_final_frac=0.1, lr_anneal_episodes=None, episodes=11)

    assert lr_scale_at(args, 1) == pytest.approx(1.0)
    assert lr_scale_at(args, 6) == pytest.approx(0.55)  # halfway: 1.0 -> 0.1
    assert lr_scale_at(args, 11) == pytest.approx(0.1)


def test_lr_scale_holds_final_after_window_and_uses_own_horizon() -> None:
    args = Namespace(lr_anneal_final_frac=0.1, lr_anneal_episodes=10, episodes=100)

    assert lr_scale_at(args, 10) == pytest.approx(0.1)
    # Clamped to the final fraction past the (shorter) anneal window.
    assert lr_scale_at(args, 80) == pytest.approx(0.1)
