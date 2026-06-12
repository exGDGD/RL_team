"""Re-evaluate a saved ACAC checkpoint at high sample size, WITHOUT retraining.

Loads a checkpoint (e.g. ``best.pt``), reconstructs the policy and the env
config it was trained with, then evaluates each workload scenario on held-out
seeds with a configurable (per-scenario) episode count. Reports per-scenario
RL vs baselines, the balanced score, and RL-vs-best-baseline significance
(Δ ± 95% CI) so a final number is trustworthy and you can tell whether the
sample is large enough (give the noisy scenarios -- e.g. burst_stress -- more).

Usage::

    python -m src.eval_checkpoint outputs/acac_flow/best.pt --episodes 50
    python -m src.eval_checkpoint outputs/acac_flow/best.pt \
        --scenario-episodes balanced=80,bg_heavy=60,burst_stress=200,ui_heavy=60 \
        --seed 30000 --out outputs/acac_flow/eval_checkpoint.json

``--episodes`` is the count *per scenario* (not the round-robin total). The base
seed defaults to 30000, disjoint from the training/eval/test seed ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.env import WorkloadScenario

_BASELINE_COLS = ("random", "mlfq", "sjf_like", "eas_like")


def parse_scenario_episodes(
    spec: str | None,
    *,
    default: int,
    scenarios: tuple[WorkloadScenario, ...],
) -> dict[WorkloadScenario, int]:
    """Episodes per scenario: ``default`` for each, overridden by ``spec``.

    ``spec`` is a comma list like ``"burst_stress=200,balanced=80"``. Lets the
    high-variance scenarios get more episodes than the easy ones.
    """
    counts = {scenario: int(default) for scenario in scenarios}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected 'name=count', got {item!r}")
        counts[WorkloadScenario(name.strip())] = int(value)
    return counts


def assemble_summary(per_scenario: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge single-scenario ``evaluate_policy`` results into one summary.

    ``per_scenario`` maps each scenario value to the result of evaluating that
    scenario alone. The merged dict has the same shape the rest of the code
    expects (``by_scenario`` / ``sampled.by_scenario`` / ``baselines.*.by_scenario``)
    so ``checkpoint_score`` and ``summarize_scenario_significance`` work on it.
    """
    rl_by: dict[str, Any] = {}
    sampled_by: dict[str, Any] = {}
    baselines_by: dict[str, dict[str, Any]] = {}
    for key, summary in per_scenario.items():
        rl_by[key] = summary["by_scenario"][key]
        sampled_by[key] = summary.get("sampled", {}).get("by_scenario", {}).get(key, {})
        for name, entry in (summary.get("baselines") or {}).items():
            row = (entry.get("by_scenario") or {}).get(key)
            if row is not None:
                baselines_by.setdefault(name, {})[key] = row
    return {
        "by_scenario": rl_by,
        "sampled": {"by_scenario": sampled_by},
        "baselines": {name: {"by_scenario": by} for name, by in baselines_by.items()},
    }


def format_report(
    summary: dict[str, Any],
    *,
    checkpoint: Any,
    meta: dict[str, Any] | None = None,
) -> str:
    """Render a copy-paste-friendly text report (table + significance)."""
    from src.plot_metrics import ORACLE_BASELINES, summarize_scenario_significance
    from src.train_acac import checkpoint_score

    meta = meta or {}
    score = checkpoint_score(summary)
    lines = [
        f"checkpoint={checkpoint} iter={meta.get('episode')} | "
        f"balanced_score={'-' if score is None else f'{score:+.3f}'} "
        "(0=best realistic baseline, >0 beats it; SJF=clairvoyant oracle ceiling, not a target)"
    ]

    def fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.1f}"

    # Mark the clairvoyant oracle column so the table is not read as a fair target.
    cols = [f"{c}*" if c in ORACLE_BASELINES else c for c in _BASELINE_COLS]
    header = f"{'scenario':<13}{'n':>5}{'rl':>10}{'rl_smp':>10}" + "".join(
        f"{col:>10}" for col in cols
    )
    lines.append(header)
    by_scenario = summary.get("by_scenario", {})
    sampled = summary.get("sampled", {}).get("by_scenario", {})
    baselines = summary.get("baselines", {})
    for name in by_scenario:
        rl = by_scenario[name]
        row = (
            f"{name:<13}{int(rl.get('n', 0)):>5}"
            f"{fmt(rl.get('reward')):>10}{fmt(sampled.get(name, {}).get('reward')):>10}"
        )
        for col in _BASELINE_COLS:
            value = baselines.get(col, {}).get("by_scenario", {}).get(name, {}).get("reward")
            row += f"{fmt(value):>10}"
        lines.append(row)

    significance = summarize_scenario_significance(summary)
    if significance:
        lines.append("")
        lines.append(
            "vs best REALISTIC baseline (Δ ± 95% CI; WIN/LOSE if resolved) | SJF oracle ceiling:"
        )
        for name in sorted(significance):
            s = significance[name]
            if s["significant"] is None:
                verdict = "?   "
            elif not s["significant"]:
                verdict = "tie "
            else:
                verdict = "WIN " if s["delta"] > 0 else "LOSE"
            oracle = (
                ""
                if s["oracle"] is None
                else f" | oracle(sjf*) {s['oracle']:+9.1f} gap {s['oracle_gap']:+8.1f}"
            )
            lines.append(
                f"  {name:<13} n={s['n']:<4} rl {s['rl']:+9.1f}±{s['rl_se']:5.1f} vs "
                f"{s['best_baseline']:<8} {s['best']:+9.1f} | Δ {s['delta']:+8.1f} ±{s['half_ci']:5.1f} {verdict}{oracle}"
            )
        lines.append("  (* SJF = clairvoyant oracle: knows job lengths; ceiling to approach, not a fair target)")
    return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    scenario_episodes: dict[WorkloadScenario, int],
    base_seed: int,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and evaluate each scenario with its own episode count."""
    from argparse import Namespace
    from dataclasses import fields

    import torch

    from src.rl.trainer import ACACConfig, TorchACACPolicy
    from src.train_acac import evaluate_policy

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config_fields = {field.name for field in fields(ACACConfig)}
    config = ACACConfig(**{k: v for k, v in ckpt["config"].items() if k in config_fields})
    policy = TorchACACPolicy(config, device=device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    # The checkpoint stores the training args (env config, reward weights), so the
    # held-out eval matches the environment the policy was trained on.
    args = Namespace(**ckpt["args"])

    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario, episodes in scenario_episodes.items():
        per_scenario[scenario.value] = evaluate_policy(
            policy,
            args,
            base_seed=base_seed,
            episodes=episodes,
            scenarios=(scenario,),
        )
    return assemble_summary(per_scenario), {"episode": ckpt.get("episode")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate an ACAC checkpoint at high sample size (no retraining)."
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path, e.g. outputs/acac_flow/best.pt")
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes PER scenario (default 50). Total = this x #scenarios.",
    )
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument(
        "--scenario-episodes",
        type=str,
        default="",
        help="Per-scenario overrides, e.g. 'burst_stress=200,balanced=80'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=30000,
        help="Held-out base seed (disjoint from train/eval/test).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the merged summary as JSON.",
    )
    args = parser.parse_args()

    from src.train_acac import parse_workload_scenarios

    scenarios = parse_workload_scenarios(args.scenarios)
    scenario_episodes = parse_scenario_episodes(
        args.scenario_episodes, default=args.episodes, scenarios=scenarios
    )

    summary, meta = evaluate_checkpoint(
        args.checkpoint,
        scenario_episodes=scenario_episodes,
        base_seed=args.seed,
        device=args.device,
    )
    print(format_report(summary, checkpoint=args.checkpoint, meta=meta))
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
