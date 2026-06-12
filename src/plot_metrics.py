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

import numpy as np


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


def latest_scenario_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-scenario eval rewards from the most recent evaluated iteration.

    Returns ``{scenario: {"rl", "rl_sampled", "random", "mlfq", "sjf_like",
    "eas_like"}}`` (values are rewards or ``None`` when absent). Empty dict when
    no row carries per-scenario eval data (single-scenario run or old log).
    """
    for row in reversed(rows):
        by_scenario = _pluck(row, "evaluation", "by_scenario")
        if not by_scenario:
            continue
        summary: dict[str, dict[str, Any]] = {}
        for scenario in sorted(by_scenario):
            summary[scenario] = {
                "rl": _pluck(row, "evaluation", "by_scenario", scenario, "reward"),
                "rl_sampled": _pluck(
                    row, "evaluation", "sampled", "by_scenario", scenario, "reward"
                ),
                **{
                    name: _pluck(
                        row, "evaluation", "baselines", name, "by_scenario", scenario, "reward"
                    )
                    for name in ("random", "mlfq", "sjf_like", "eas_like")
                },
            }
        return summary
    return {}


def _clip_floor(
    ax,
    values: list[Any],
    override: float | None = None,
    *,
    keep_visible: list[Any] = (),
) -> None:
    """Clip a reward axis's lower bound so a few very-negative eval points don't
    compress the meaningful range.

    A deterministic policy that completes nothing scores an enormous negative
    reward (e.g. -14000 vs the usual -700..-3000), which otherwise flattens
    every other curve. ``override`` sets an explicit floor; without it the floor
    is the 5th percentile of the plotted values (so a small fraction of extreme
    outliers clip off the bottom while the bulk stays visible). ``keep_visible``
    values (e.g. baseline reference lines) are never clipped.
    """
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if override is not None:
        ax.set_ylim(bottom=override)
        return
    if len(finite) < 4:
        return
    floor = float(np.percentile(finite, 5))
    keep = [float(v) for v in keep_visible if v is not None and np.isfinite(v)]
    if keep:
        floor = min(floor, min(keep))
    top = max(finite + keep)
    pad = 0.03 * (top - floor) if top > floor else (abs(floor) * 0.03 or 1.0)
    ax.set_ylim(bottom=floor - pad)


def _reward_se(row: dict[str, Any]) -> float:
    """Standard error of a scenario's mean reward: std / sqrt(n) (0 if absent)."""
    n = row.get("n") or 0
    std = row.get("reward_std")
    if not n or std is None:
        return 0.0
    return float(std) / float(np.sqrt(n))


def summarize_scenario_significance(
    summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Per-scenario RL-vs-best-baseline reward gap with standard errors.

    For judging whether an eval/test sample is large enough: the gap ``Δ = rl -
    best_baseline`` is only trustworthy when its 95% half-interval
    (``1.96 * combined_SE``) is smaller than ``|Δ|``. ``SE = reward_std/sqrt(n)``
    for each side, combined in quadrature. ``significant`` is True when
    ``|Δ| > 1.96*SE`` (the sample resolves the difference), False when it does
    not (need more episodes for that scenario), and ``None`` when SE is 0.
    Returns ``{}`` when std/n are missing (older logs without them).
    """
    by_scenario = (summary or {}).get("by_scenario") or {}
    baselines = (summary or {}).get("baselines") or {}
    out: dict[str, dict[str, Any]] = {}
    for scenario, row in by_scenario.items():
        rl_mean = row.get("reward")
        if rl_mean is None or "n" not in row:
            continue
        best_name, best_row = None, None
        for name, entry in baselines.items():
            brow = (entry.get("by_scenario") or {}).get(scenario)
            if brow and brow.get("reward") is not None:
                if best_row is None or brow["reward"] > best_row["reward"]:
                    best_name, best_row = name, brow
        if best_row is None:
            continue
        delta = float(rl_mean) - float(best_row["reward"])
        se = float(np.sqrt(_reward_se(row) ** 2 + _reward_se(best_row) ** 2))
        half_ci = 1.96 * se
        out[scenario] = {
            "n": int(row.get("n") or 0),
            "rl": float(rl_mean),
            "rl_se": _reward_se(row),
            "best_baseline": best_name,
            "best": float(best_row["reward"]),
            "best_se": _reward_se(best_row),
            "delta": delta,
            "half_ci": half_ci,
            "significant": (abs(delta) > half_ci) if se > 0 else None,
        }
    return out


def plot_training_metrics(
    rows: list[dict[str, Any]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
    reward_floor: float | None = None,
):
    """Render a 4x2 panel of training curves and optionally save/show it.

    ``reward_floor`` sets an explicit lower y-limit on the reward panels (the
    aggregate one and the two per-scenario ones); when ``None`` a robust
    automatic floor clips off extreme eval spikes (see ``_clip_floor``).

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

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
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
    baselines = _latest_baselines(rows)
    for name, color in (
        ("sjf_like", "C2"),
        ("eas_like", "C3"),
        ("mlfq", "C5"),
        ("random", "C4"),
    ):
        reward = baselines.get(name)
        if reward is not None:
            ax.axhline(reward, ls="--", lw=1, color=color, label=f"{name} {reward:.0f}")
    ax.set_title("reward (higher = better)")
    ax.legend(fontsize=8, loc="best")
    _clip_floor(ax, train_y + eval_y + samp_y, reward_floor, keep_visible=list(baselines.values()))

    # Overlay the scenario-balanced checkpoint score on a right axis: the honest
    # convergence/early-stop signal that the burst-dominated aggregate reward
    # hides (a flat aggregate can mask a policy that peaks then drifts).
    score_x, score_y = _series(rows, "evaluation", "balanced_score")
    if score_x:
        ax_score = ax.twinx()
        ax_score.plot(score_x, score_y, "s-", ms=3, color="k", lw=1.3, label="balanced_score")
        ax_score.axhline(0.0, ls=":", lw=0.8, color="0.6")  # 0 = matched best baseline
        ax_score.set_ylabel("balanced_score (0=best baseline, <0 behind)")
        ax_score.legend(loc="lower right", fontsize=7)

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

    # --- Per-scenario eval reward (deterministic + sampled) ------------------
    scenarios = sorted(
        {
            key
            for row in rows
            for key in (_pluck(row, "evaluation", "by_scenario") or {})
        }
    )
    for ax, path, label in (
        (axes[3, 0], ("evaluation", "by_scenario"), "deterministic"),
        (axes[3, 1], ("evaluation", "sampled", "by_scenario"), "sampled"),
    ):
        panel_ys: list[Any] = []
        for i, scenario in enumerate(scenarios):
            xs, ys = _series(rows, *path, scenario, "reward")
            if xs:
                ax.plot(xs, ys, "o-", ms=3, color=f"C{i}", label=scenario)
                panel_ys.extend(ys)
        ax.set_title(f"eval reward by scenario ({label})")
        if scenarios:
            ax.legend(fontsize=7)
        _clip_floor(ax, panel_ys, reward_floor)

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
    parser.add_argument(
        "--reward-floor",
        type=float,
        default=None,
        help=(
            "Explicit lower y-limit for the reward panels. Default clips a "
            "robust automatic floor so extreme eval spikes don't compress the "
            "curves."
        ),
    )
    args = parser.parse_args()

    out = args.out or args.metrics.with_name("metrics.png")
    rows = load_metrics(args.metrics)
    plot_training_metrics(
        rows,
        save_path=out,
        show=args.show,
        title=str(args.metrics),
        reward_floor=args.reward_floor,
    )
    print(f"saved plot={out} (iterations={len(rows)})")


if __name__ == "__main__":
    main()
