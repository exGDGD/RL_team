import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for the tests

from src.plot_metrics import load_metrics, plot_training_metrics


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
            "sampled": {
                "reward": -735.0,
                "by_scenario": {
                    "balanced": {"reward": -740.0},
                    "ui_heavy": {"reward": -745.0},
                },
            },
            "baselines": {
                "sjf_like": {"reward": -745.9},
                "eas_like": {"reward": -848.2},
                "mlfq": {"reward": -800.0},
                "random": {"reward": -955.8},
            },
            "by_scenario": {
                "balanced": {"reward": -700.0},
                "ui_heavy": {"reward": -720.0},
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
