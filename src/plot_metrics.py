"""Plot training curves from a ``metrics.jsonl`` produced by ``train_acac``.

Usage
-----
CLI (saves a PNG next to the log)::

    python -m src.plot_metrics outputs/acac_flow/metrics.jsonl
    python -m src.plot_metrics outputs/acac_flow/metrics.jsonl --out curves.png --show

Notebook (inline)::

    from src.plot_metrics import load_metrics, plot_training_metrics
    plot_training_metrics(load_metrics("outputs/acac_flow/metrics.jsonl"), show=True)

Each row in the log is one training *iteration* (a merged batch of
``--rollout-episodes`` episodes), so the x-axis is the iteration index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL metrics file into a list of row dicts (blank lines skipped)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pluck(row: dict[str, Any], *path: str) -> Any:
    """Return ``row[path[0]][path[1]]...`` or ``None`` if any step is missing."""
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _series(
    rows: list[dict[str, Any]],
    *path: str,
    x: str = "episode",
) -> tuple[list[Any], list[Any]]:
    """Collect (x, y) pairs for the rows that actually contain ``path``.

    Rows missing the value (e.g. eval-only fields on non-eval iterations, or
    fields absent in logs from older code) are skipped, so partial logs still
    plot what they have.
    """
    xs: list[Any] = []
    ys: list[Any] = []
    for row in rows:
        value = _pluck(row, *path)
        if value is None:
            continue
        xs.append(row.get(x))
        ys.append(value)
    return xs, ys


def _latest_baselines(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Baseline rewards are constant across a run; grab the last eval's copy."""
    for row in reversed(rows):
        baselines = _pluck(row, "evaluation", "baselines")
        if baselines:
            return {
                name: entry.get("reward")
                for name, entry in baselines.items()
                if isinstance(entry, dict) and entry.get("reward") is not None
            }
    return {}


def plot_training_metrics(
    rows: list[dict[str, Any]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Render a 3x2 panel of training curves and optionally save/show it.

    Returns the matplotlib Figure so callers (e.g. notebooks) can tweak it.
    """
    if not rows:
        raise ValueError("No metrics rows to plot.")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib` (it is preinstalled on Colab)."
        ) from exc

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    if title:
        fig.suptitle(title)

    # --- Reward vs baselines -------------------------------------------------
    ax = axes[0, 0]
    train_x, train_y = _series(rows, "reward")
    ax.plot(train_x, train_y, color="0.7", lw=1, label="train (per-iter avg)")
    eval_x, eval_y = _series(rows, "evaluation", "reward")
    ax.plot(eval_x, eval_y, "o-", ms=3, color="C0", label="eval (deterministic)")
    samp_x, samp_y = _series(rows, "evaluation", "sampled", "reward")
    ax.plot(samp_x, samp_y, "o-", ms=3, color="C1", alpha=0.7, label="eval (sampled)")
    for name, color in (("sjf_like", "C2"), ("eas_like", "C3"), ("random", "C4")):
        reward = _latest_baselines(rows).get(name)
        if reward is not None:
            ax.axhline(reward, ls="--", lw=1, color=color, label=f"{name} {reward:.0f}")
    ax.set_title("reward (higher = better)")
    ax.legend(fontsize=8, loc="best")

    # --- Loss components -----------------------------------------------------
    ax = axes[0, 1]
    for key, label in (("loss", "loss"), ("policy_loss", "policy"), ("value_loss", "value")):
        xs, ys = _series(rows, "update", key)
        if xs:
            ax.plot(xs, ys, lw=1, label=label)
    ax.axhline(0.0, ls=":", lw=0.8, color="0.5")
    ax.set_title("loss")
    ax.legend(fontsize=8)

    # --- Entropy and its (annealed) coefficient ------------------------------
    ax = axes[1, 0]
    ent_x, ent_y = _series(rows, "update", "entropy")
    ax.plot(ent_x, ent_y, color="C0", lw=1, label="entropy (nats)")
    ax.set_ylabel("entropy", color="C0")
    coef_x, coef_y = _series(rows, "update", "entropy_coef")
    if coef_x:
        ax2 = ax.twinx()
        ax2.plot(coef_x, coef_y, color="C3", lw=1, ls="--", label="entropy_coef")
        ax2.set_ylabel("entropy_coef", color="C3")
    ax.set_title("entropy & coefficient")

    # --- KL and clip fraction ------------------------------------------------
    ax = axes[1, 1]
    kl_x, kl_y = _series(rows, "update", "approx_kl")
    ax.plot(kl_x, kl_y, color="C0", lw=1, label="approx_kl")
    ax.set_ylabel("approx_kl", color="C0")
    clip_x, clip_y = _series(rows, "update", "clip_fraction")
    if clip_x:
        ax2 = ax.twinx()
        ax2.plot(clip_x, clip_y, color="C1", lw=1, label="clip_fraction")
        ax2.set_ylabel("clip_fraction", color="C1")
    ax.set_title("KL & clip fraction")

    # --- Gradient norms ------------------------------------------------------
    ax = axes[2, 0]
    for key, label in (("actor_grad_norm", "actor"), ("critic_grad_norm", "critic")):
        xs, ys = _series(rows, "update", key)
        if xs:
            ax.plot(xs, ys, lw=1, label=label)
    ax.set_title("gradient norm")
    ax.legend(fontsize=8)

    # --- Decision mix --------------------------------------------------------
    ax = axes[2, 1]
    for key, label in (
        ("noop_fraction", "no-op fraction"),
        ("forced_decision_fraction", "forced fraction"),
    ):
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, lw=1, label=label)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("decision mix")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train_acac metrics.jsonl curves.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: metrics.png next to the input).",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure window.")
    args = parser.parse_args()

    out = args.out or args.metrics.with_name("metrics.png")
    rows = load_metrics(args.metrics)
    plot_training_metrics(rows, save_path=out, show=args.show, title=str(args.metrics))
    print(f"saved plot={out} (iterations={len(rows)})")


if __name__ == "__main__":
    main()
