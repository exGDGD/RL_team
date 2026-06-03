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
    """One compact progress line emitted every training episode."""

    get_logger().info(
        "ep %4d | reward %+8.3f | loss %7.3f (pi %+.4f  v %.3f  ent %.3f) "
        "| kl %.4f clip %.3f | grad a/c %.2f/%.2f "
        "| conflict %d choices %.2f forced %.2f "
        "| done %d/%d thru %.3f turn %s",
        episode_idx,
        total_reward,
        stats.loss,
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.approx_kl,
        stats.clip_fraction,
        stats.actor_grad_norm,
        stats.critic_grad_norm,
        rollout.conflicts,
        rollout.mean_task_choices,
        rollout.forced_decision_fraction,
        metrics.completed_tasks,
        metrics.total_tasks,
        metrics.throughput,
        _fmt(metrics.mean_turnaround_time),
    )


def log_episode_eval(eval_summary: dict[str, Any]) -> None:
    """Indented evaluation block emitted only on eval episodes."""

    logger = get_logger()
    baselines = eval_summary.get("baselines", {})
    logger.info(
        "       eval | reward %+.3f (sampled %+.3f) | done %.1f "
        "| first_slot %.2f/%.2f",
        eval_summary["reward"],
        eval_summary["sampled"]["reward"],
        eval_summary["completed"],
        eval_summary["actions"]["first_slot_fraction"],
        eval_summary["sampled"]["actions"]["first_slot_fraction"],
    )
    logger.info(
        "       base | random %+.3f  sjf %+.3f  eas %+.3f",
        _baseline_reward(baselines, "random"),
        _baseline_reward(baselines, "sjf_like"),
        _baseline_reward(baselines, "eas_like"),
    )


def _baseline_reward(baselines: dict[str, Any], name: str) -> float:
    entry = baselines.get(name)
    if not entry or entry.get("reward") is None:
        return float("nan")
    return float(entry["reward"])


def _fmt(value: float | None, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    return format(value, spec)
