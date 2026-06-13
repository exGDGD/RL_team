import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display needed for the tests

from src.plot_metrics import (
    latest_eval_summary,
    latest_scenario_summary,
    load_metrics,
    plot_training_metrics,
    scenario_metric_table,
    summarize_scenario_significance,
)


def _row(episode: int, *, eval_row: bool) -> dict:
    row = {
        "episode": episode,
        "reward": -700.0 - episode,
        "noop_fraction": 0.6,
        "forced_decision_fraction": 0.3,
        "update": {
            "loss": 0.01,
            "policy_loss": -0.005,
            "value_loss": 0.01,
            "entropy": 0.6,
            "entropy_coef": 0.01 * (1 - episode / 10),
            "approx_kl": 0.01,
            "clip_fraction": 0.07,
            "actor_grad_norm": 0.4,
            "critic_grad_norm": 0.05,
        },
    }
    row["evaluation"] = (
        {
            "reward": -720.0,
            "balanced_score": 0.5 - episode * 0.03,
            "sampled": {
                "reward": -735.0,
                "by_scenario": {
                    "balanced": {"reward": -740.0},
                    "ui_heavy": {"reward": -745.0},
                },
            },
            "baselines": {
                "sjf_like": {
                    "reward": -745.9,
                    "by_scenario": {"balanced": {"reward": -740.0}, "ui_heavy": {"reward": -750.0}},
                },
                "eas_like": {
                    "reward": -848.2,
                    "by_scenario": {"balanced": {"reward": -840.0}, "ui_heavy": {"reward": -855.0}},
                },
                "mlfq": {
                    "reward": -800.0,
                    "by_scenario": {"balanced": {"reward": -795.0}, "ui_heavy": {"reward": -805.0}},
                },
                "random": {
                    "reward": -955.8,
                    "by_scenario": {"balanced": {"reward": -950.0}, "ui_heavy": {"reward": -960.0}},
                },
            },
            "by_scenario": {
                "balanced": {"reward": -700.0, "turnaround": 12.0, "throughput": 1.5},
                "ui_heavy": {"reward": -720.0, "turnaround": 14.0, "throughput": 1.2},
            },
        }
        if eval_row
        else None
    )
    return row


def test_load_metrics_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(_row(1, eval_row=True)) + "\n\n" + json.dumps(_row(2, eval_row=False)) + "\n",
        encoding="utf-8",
    )

    rows = load_metrics(path)

    assert [r["episode"] for r in rows] == [1, 2]


def test_plot_training_metrics_writes_png(tmp_path: Path) -> None:
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 11)]
    out = tmp_path / "curves.png"

    fig = plot_training_metrics(rows, save_path=out, title="test run")

    assert out.exists() and out.stat().st_size > 0
    # 4x2 grid of panels (plus twin axes for entropy/KL).
    assert len(fig.axes) >= 8


def test_plot_reward_floor_clips_extreme_eval(tmp_path: Path) -> None:
    """A single degenerate deterministic-eval spike (e.g. completes nothing)
    must not drag the reward axis down and compress every other curve."""
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 11)]
    spike = _row(11, eval_row=True)
    spike["evaluation"]["reward"] = -14000.0  # normal range is ~ -700..-955
    rows.append(spike)

    fig = plot_training_metrics(rows, save_path=tmp_path / "c.png")
    bottom = fig.axes[0].get_ylim()[0]

    assert bottom > -3000.0  # the -14000 spike is clipped off the bottom
    assert bottom < -955.0  # the worst baseline (random -955.8) stays visible


def test_plot_overlays_balanced_score_axis(tmp_path: Path) -> None:
    """The reward panel gains a right-axis balanced_score curve when present."""
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 6)]

    fig = plot_training_metrics(rows, save_path=tmp_path / "c.png")

    assert any("balanced_score" in ax.get_ylabel() for ax in fig.axes)


def test_plot_without_balanced_score_has_no_score_axis(tmp_path: Path) -> None:
    rows = [_row(i, eval_row=True) for i in range(1, 4)]
    for row in rows:
        row["evaluation"].pop("balanced_score")

    fig = plot_training_metrics(rows, save_path=tmp_path / "c.png")

    assert not any("balanced_score" in ax.get_ylabel() for ax in fig.axes)


def test_plot_reward_floor_explicit_override(tmp_path: Path) -> None:
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 6)]

    fig = plot_training_metrics(rows, save_path=tmp_path / "c.png", reward_floor=-2000.0)

    assert fig.axes[0].get_ylim()[0] == pytest.approx(-2000.0)


def test_latest_scenario_summary_extracts_rl_and_baselines() -> None:
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 5)]

    summary = latest_scenario_summary(rows)

    assert set(summary) == {"balanced", "ui_heavy"}
    bal = summary["balanced"]
    assert bal["rl"] == pytest.approx(-700.0)
    assert bal["rl_sampled"] == pytest.approx(-740.0)
    assert bal["mlfq"] == pytest.approx(-795.0)
    assert bal["sjf_like"] == pytest.approx(-740.0)
    assert bal["eas_like"] == pytest.approx(-840.0)
    assert bal["random"] == pytest.approx(-950.0)


def test_summarize_scenario_significance_compares_realistic_and_reports_oracle() -> None:
    summary = {
        "by_scenario": {
            "x": {"reward": -100.0, "reward_std": 8.0, "n": 4},   # SE = 4.0
            "y": {"reward": -100.0, "reward_std": 20.0, "n": 4},  # SE = 10.0
        },
        "baselines": {
            "mlfq": {  # best realistic competitor
                "by_scenario": {
                    "x": {"reward": -90.0, "reward_std": 4.0, "n": 16},  # SE = 1.0
                    "y": {"reward": -95.0, "reward_std": 4.0, "n": 16},
                }
            },
            "sjf_like": {  # clairvoyant oracle -> reported as ceiling, not the competitor
                "by_scenario": {
                    "x": {"reward": -70.0, "reward_std": 2.0, "n": 16},
                    "y": {"reward": -80.0, "reward_std": 2.0, "n": 16},
                }
            },
        },
    }

    sig = summarize_scenario_significance(summary)

    assert sig["x"]["best_baseline"] == "mlfq"           # not sjf_like
    assert sig["x"]["delta"] == pytest.approx(-10.0)     # vs mlfq, not the oracle
    assert sig["x"]["significant"] is True               # |-10| > 1.96*sqrt(16+1)=8.08
    assert sig["x"]["oracle"] == pytest.approx(-70.0)
    assert sig["x"]["oracle_gap"] == pytest.approx(-30.0)  # rl - oracle
    assert sig["y"]["significant"] is False              # |-5| < 1.96*sqrt(100+1)=19.7


def test_summarize_scenario_significance_empty_without_realistic_baseline() -> None:
    # Only the oracle present -> no fair competitor -> nothing to report.
    summary = {
        "by_scenario": {"x": {"reward": -100.0, "reward_std": 4.0, "n": 8}},
        "baselines": {"sjf_like": {"by_scenario": {"x": {"reward": -90.0, "n": 8, "reward_std": 2.0}}}},
    }
    assert summarize_scenario_significance(summary) == {}


def test_summarize_scenario_significance_empty_without_std() -> None:
    summary = {
        "by_scenario": {"x": {"reward": -100.0}},  # no n / reward_std (old log)
        "baselines": {"mlfq": {"by_scenario": {"x": {"reward": -90.0}}}},
    }
    assert summarize_scenario_significance(summary) == {}


def test_latest_scenario_summary_empty_without_scenario_data() -> None:
    rows = [_row(i, eval_row=False) for i in range(1, 3)]
    assert latest_scenario_summary(rows) == {}


def test_latest_eval_summary_returns_last_eval_block() -> None:
    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 6)]
    ev = latest_eval_summary(rows)
    assert ev is not None and "by_scenario" in ev
    # Most recent *evaluated* iteration (i=5 here).
    assert ev["by_scenario"]["balanced"]["reward"] == -700.0


def test_latest_eval_summary_none_without_eval() -> None:
    rows = [_row(i, eval_row=False) for i in range(1, 4)]
    assert latest_eval_summary(rows) is None


def test_scenario_metric_table_reads_arbitrary_metric() -> None:
    summary = {
        "by_scenario": {"balanced": {"reward": -700.0, "turnaround": 12.0}},
        "sampled": {"by_scenario": {"balanced": {"reward": -740.0, "turnaround": 13.5}}},
        "baselines": {
            "mlfq": {"by_scenario": {"balanced": {"reward": -800.0, "turnaround": 15.0}}},
            "sjf_like": {"by_scenario": {"balanced": {"reward": -745.0, "turnaround": 9.0}}},
        },
    }

    table = scenario_metric_table(summary, "turnaround")

    assert table["balanced"]["rl"] == 12.0
    assert table["balanced"]["rl_sampled"] == 13.5
    assert table["balanced"]["mlfq"] == 15.0
    assert table["balanced"]["sjf_like"] == 9.0
    assert table["balanced"]["random"] is None  # baseline absent -> None, not a KeyError


def test_plot_training_metrics_interactive_smoke(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    from src.plot_metrics import plot_training_metrics_interactive

    rows = [_row(i, eval_row=(i % 2 == 1)) for i in range(1, 8)]
    out = tmp_path / "curves.html"

    fig = plot_training_metrics_interactive(rows, show=False, save_html=out, title="t")

    assert out.exists() and out.stat().st_size > 0
    assert len(fig.data) > 0  # traces were added (reward/loss/per-scenario/turnaround/...)


def test_plot_handles_rows_without_new_fields(tmp_path: Path) -> None:
    """Logs from older code lack entropy_coef/noop_fraction/eval — plotting must
    still succeed, drawing only the series that are present."""
    rows = []
    for i in range(1, 5):
        row = _row(i, eval_row=False)
        row.pop("noop_fraction")
        row["update"].pop("entropy_coef")
        rows.append(row)
    out = tmp_path / "curves_old.png"

    plot_training_metrics(rows, save_path=out)

    assert out.exists() and out.stat().st_size > 0
