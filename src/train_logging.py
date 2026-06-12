"""Console/file logging helpers for training scripts.

The structured, machine-readable record still goes to ``metrics.jsonl`` via
``train_acac.append_jsonl``. This module only handles the *human-readable*
console (and optional ``train.log`` file) output so training progress is easy
to scan at a glance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "acac.train"


def configure_logging(
    output_dir: Path | None = None,
    *,
    level: int = logging.INFO,
    log_filename: str = "train.log",
) -> logging.Logger:
    """Set up a console (and optional file) logger for training.

    Safe to call more than once (e.g. Colab cell re-runs): existing handlers
    are cleared so logs are not duplicated. When ``output_dir`` is given, the
    same lines are mirrored to ``output_dir/train.log``.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the shared training logger (configure_logging sets it up)."""

    return logging.getLogger(LOGGER_NAME)


def log_episode_train(
    *,
    episode_idx: int,
    total_reward: float,
    rollout: Any,
    metrics: Any,
    stats: Any,
) -> None:
    """One compact progress line emitted every training update iteration."""

    preempt_per_episode = rollout.preemptions / max(rollout.episodes, 1)
    entropy_coef = getattr(stats, "entropy_coef", 0.0)
    get_logger().info(
        "iter %4d | reward %+8.3f | loss %7.3f (pi %+.4f  v %.3f  ent %.3f ec %.4f) "
        "| kl %.4f clip %.3f | grad a/c %.2f/%.2f "
        "| trans %d/%d noop %d (%.2f) conflict %d preempt %.1f choices %.2f forced %.2f "
        "| done %.1f/%.1f thru %.3f turn %s",
        episode_idx,
        total_reward,
        stats.loss,
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        entropy_coef,
        stats.approx_kl,
        stats.clip_fraction,
        stats.actor_grad_norm,
        stats.critic_grad_norm,
        len(rollout.transitions),
        len(rollout.joint_transitions),
        rollout.noop_decisions,
        rollout.noop_fraction,
        rollout.conflicts,
        preempt_per_episode,
        rollout.mean_task_choices,
        rollout.forced_decision_fraction,
        metrics["completed_tasks"],
        metrics["total_tasks"],
        metrics["throughput"],
        _fmt(metrics["mean_turnaround_time"]),
    )


def log_episode_eval(eval_summary: dict[str, Any]) -> None:
    """Indented evaluation block emitted only on eval episodes."""

    logger = get_logger()
    baselines = eval_summary.get("baselines", {})
    logger.info(
        "       eval | reward %+.3f (sampled %+.3f) | score %s | done %.1f | preempt %.1f "
        "| first_slot %.2f/%.2f",
        eval_summary["reward"],
        eval_summary["sampled"]["reward"],
        _fmt(eval_summary.get("balanced_score")),
        eval_summary["completed"],
        eval_summary.get("preemptions", float("nan")),
        eval_summary["actions"]["first_slot_fraction"],
        eval_summary["sampled"]["actions"]["first_slot_fraction"],
    )
    logger.info(
        "       base | random %+.3f  mlfq %+.3f  sjf %+.3f  eas %+.3f",
        _baseline_reward(baselines, "random"),
        _baseline_reward(baselines, "mlfq"),
        _baseline_reward(baselines, "sjf_like"),
        _baseline_reward(baselines, "eas_like"),
    )
    for scenario in sorted(eval_summary.get("by_scenario", {})):
        row = eval_summary["by_scenario"][scenario]
        sampled_row = eval_summary.get("sampled", {}).get("by_scenario", {}).get(scenario, {})
        logger.info(
            "       eval/%s | reward %+.3f (sampled %+.3f) | base r/m/s/e %+.3f/%+.3f/%+.3f/%+.3f",
            scenario,
            _row_value(row, "reward"),
            _row_value(sampled_row, "reward"),
            _baseline_scenario_reward(baselines, "random", scenario),
            _baseline_scenario_reward(baselines, "mlfq", scenario),
            _baseline_scenario_reward(baselines, "sjf_like", scenario),
            _baseline_scenario_reward(baselines, "eas_like", scenario),
        )


def log_scenario_significance(significance: dict[str, Any]) -> None:
    """Per-scenario RL-vs-best-baseline gap with 95% CI, to judge sample size.

    ``sig=YES`` means the sample resolves the gap (|Δ| > 95% half-interval);
    ``NO`` means it does not -- that scenario needs more eval/test episodes
    before the comparison can be trusted.
    """

    if not significance:
        return
    logger = get_logger()
    logger.info("       signif | RL vs best baseline (Δ ± 95% CI; sig=YES if |Δ|>CI -> sample resolves it)")
    for scenario in sorted(significance):
        s = significance[scenario]
        mark = "?" if s["significant"] is None else ("YES" if s["significant"] else "NO ")
        logger.info(
            "       signif/%-12s n=%-3d | rl %+8.1f ±%5.1f vs %-8s %+8.1f | Δ %+7.1f ± %5.1f | sig=%s",
            scenario,
            s["n"],
            s["rl"],
            s["rl_se"],
            s["best_baseline"],
            s["best"],
            s["delta"],
            s["half_ci"],
            mark,
        )


def _baseline_reward(baselines: dict[str, Any], name: str) -> float:
    entry = baselines.get(name)
    if not entry or entry.get("reward") is None:
        return float("nan")
    return float(entry["reward"])


def _baseline_scenario_reward(
    baselines: dict[str, Any],
    name: str,
    scenario: str,
) -> float:
    entry = baselines.get(name, {})
    by_scenario = entry.get("by_scenario", {})
    return _row_value(by_scenario.get(scenario, {}), "reward")


def _row_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    return format(value, spec)
