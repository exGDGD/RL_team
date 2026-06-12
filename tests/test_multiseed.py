import pytest

from src.multiseed import (
    aggregate_runs,
    compare_configs,
    format_multiseed_report,
    t95,
)

# Baselines are fixed by the eval seed -> constant across training seeds.
_BASE = {
    "random": {"balanced": -900.0, "burst_stress": -4700.0},
    "mlfq": {"balanced": -830.0, "burst_stress": -4000.0},      # best realistic
    "sjf_like": {"balanced": -760.0, "burst_stress": -3590.0},  # clairvoyant oracle
}


def _summary(rl: dict[str, float]) -> dict:
    return {
        "by_scenario": {sc: {"reward": r, "n": 50, "reward_std": 12.0} for sc, r in rl.items()},
        "baselines": {
            name: {
                "by_scenario": {sc: {"reward": _BASE[name][sc], "n": 50, "reward_std": 6.0} for sc in rl}
            }
            for name in _BASE
        },
    }


def _runs(config: str, balanced: list[float], burst: list[float]) -> list[dict]:
    return [
        {"config": config, "seed": i, "summary": _summary({"balanced": b, "burst_stress": u})}
        for i, (b, u) in enumerate(zip(balanced, burst))
    ]


def test_t95_small_samples_use_t_not_normal() -> None:
    assert t95(3) == pytest.approx(4.303)   # n=3 -> df=2
    assert t95(4) == pytest.approx(3.182)   # n=4 -> df=3
    assert t95(100) == pytest.approx(1.96)  # large -> normal


def test_aggregate_runs_uses_realistic_baseline_and_reports_oracle() -> None:
    runs = _runs("scratch", [-800.0, -795.0, -805.0], [-3600.0, -3620.0, -3580.0])

    agg = aggregate_runs(runs)
    s = agg["scratch"]["balanced"]

    assert s["n_seeds"] == 3
    assert s["rl_mean"] == pytest.approx(-800.0)
    assert s["best_baseline"] == "mlfq"          # not the sjf oracle
    assert s["best"] == pytest.approx(-830.0)
    assert s["delta"] == pytest.approx(30.0)     # RL beats mlfq by 30
    assert s["win"] is True                       # 30 > 95% CI across seeds
    assert s["oracle"] == pytest.approx(-760.0)
    assert s["oracle_gap"] == pytest.approx(-40.0)


def test_aggregate_runs_win_false_when_within_seed_noise() -> None:
    # Wide spread across seeds -> the +30 gap is not resolved.
    runs = _runs("scratch", [-750.0, -800.0, -890.0], [-3600.0, -3620.0, -3580.0])
    s = aggregate_runs(runs)["scratch"]["balanced"]
    assert s["delta"] == pytest.approx((-750 - 800 - 890) / 3 - (-830.0))
    assert s["win"] is False


def test_compare_configs_resolves_warm_vs_scratch_delta() -> None:
    runs = _runs("scratch", [-800.0, -795.0, -805.0], [-3600.0, -3620.0, -3580.0])
    runs += _runs("warm", [-780.0, -785.0, -775.0], [-3590.0, -3600.0, -3580.0])

    cmp = compare_configs(runs, "scratch", "warm")

    assert cmp["balanced"]["delta"] == pytest.approx(20.0)  # warm - scratch
    assert cmp["balanced"]["resolved"] is True


def test_format_multiseed_report_smoke() -> None:
    runs = _runs("scratch", [-800.0, -795.0, -805.0], [-3600.0, -3620.0, -3580.0])
    runs += _runs("warm", [-780.0, -785.0, -775.0], [-3590.0, -3600.0, -3580.0])

    report = format_multiseed_report(runs, seeds=[0, 1, 2])

    assert "multi-seed summary" in report
    assert "balanced_score" in report
    assert "scratch" in report and "warm" in report
    assert "oracle" in report
    assert "vs scratch" in report  # config comparison block
