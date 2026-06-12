from src.env import WorkloadScenario
from src.eval_checkpoint import (
    assemble_summary,
    format_report,
    parse_scenario_episodes,
)


def test_parse_scenario_episodes_default_and_override() -> None:
    scenarios = (WorkloadScenario.BALANCED, WorkloadScenario.BURST_STRESS)

    out = parse_scenario_episodes("burst_stress=200", default=50, scenarios=scenarios)

    assert out == {
        WorkloadScenario.BALANCED: 50,
        WorkloadScenario.BURST_STRESS: 200,
    }


def test_parse_scenario_episodes_empty_uses_default() -> None:
    scenarios = (WorkloadScenario.BALANCED,)
    assert parse_scenario_episodes("", default=30, scenarios=scenarios) == {
        WorkloadScenario.BALANCED: 30
    }


def _fake_scenario_result(name: str, rl: float, baselines: dict[str, float]) -> dict:
    return {
        "by_scenario": {name: {"reward": rl, "n": 10, "reward_std": 5.0}},
        "sampled": {"by_scenario": {name: {"reward": rl - 5.0}}},
        "baselines": {
            bname: {"by_scenario": {name: {"reward": value, "n": 10, "reward_std": 3.0}}}
            for bname, value in baselines.items()
        },
    }


def test_assemble_summary_merges_scenarios() -> None:
    per_scenario = {
        "balanced": _fake_scenario_result("balanced", -800.0, {"sjf_like": -720.0, "random": -900.0}),
        "burst_stress": _fake_scenario_result(
            "burst_stress", -3800.0, {"sjf_like": -3830.0, "random": -5000.0}
        ),
    }

    merged = assemble_summary(per_scenario)

    assert set(merged["by_scenario"]) == {"balanced", "burst_stress"}
    assert merged["sampled"]["by_scenario"]["balanced"]["reward"] == -805.0
    assert (
        merged["baselines"]["sjf_like"]["by_scenario"]["burst_stress"]["reward"] == -3830.0
    )


def test_format_report_contains_score_table_and_significance() -> None:
    merged = assemble_summary(
        {"balanced": _fake_scenario_result("balanced", -800.0, {"sjf_like": -720.0, "random": -900.0})}
    )

    report = format_report(merged, checkpoint="best.pt", meta={"episode": 180})

    assert "balanced_score" in report
    assert "iter=180" in report
    assert "balanced" in report
    # std/n present -> significance section rendered.
    assert "significance" in report
