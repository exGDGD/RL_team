"""Multi-seed runner for ACAC: train across several seeds (scratch and/or warm
start), re-evaluate each best.pt at high sample on held-out seeds, and aggregate
across seeds so a result is reproducible, not a single lucky run.

Analysis is oracle-aware: SJF is clairvoyant (knows job lengths) so it is the
ceiling, never the target. The headline per scenario is RL vs the best
*realistic* baseline (random/mlfq/eas), with the SJF oracle gap reported
separately.

Usage::

    # train scratch+warm over 3 seeds, then aggregate (warm needs the SJF
    # imitation actors -- run `python -m src.train_sjf_imitation ...` first):
    python -m src.multiseed --seeds 0,1,2 --configs scratch,warm \
        --episodes 300 \
        --eval-scenario-episodes balanced=80,bg_heavy=80,burst_stress=200,ui_heavy=80 \
        --out outputs/multiseed

    # re-aggregate without retraining (uses existing best.pt under --out):
    python -m src.multiseed --seeds 0,1,2 --configs scratch,warm --skip-train --out outputs/multiseed

Each (config, seed) trains into ``{out}/{config}_seed{seed}``; existing best.pt
is reused unless ``--force``. Long-running -- runs full trainings sequentially.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# 95% two-sided t-multipliers (df = n_seeds-1); ~1.96 for large n. Multi-seed
# counts are tiny, so the normal 1.96 would be anti-conservative.
_T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t95(n_samples: int) -> float:
    return _T95.get(max(n_samples - 1, 1), 1.96)


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-run summaries across seeds, per config and scenario.

    ``runs`` is a list of ``{"config", "seed", "summary"}`` (summary = a merged
    eval_checkpoint summary). Returns ``{config: {scenario: stats}}`` where stats
    has the across-seed RL mean/std/SE, the best *realistic* baseline and whether
    RL beats it (resolved across seeds), and the SJF oracle gap. Baselines are
    fixed by the eval seed, so they are read once (constant across training seeds).
    """
    import numpy as np

    from src.plot_metrics import ORACLE_BASELINES

    by_config: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_config.setdefault(run["config"], []).append(run["summary"])

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for config, summaries in by_config.items():
        scenarios = list(summaries[0].get("by_scenario", {}))
        baselines = summaries[0].get("baselines", {})
        per_scenario: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            rl_vals = [
                s["by_scenario"][scenario]["reward"]
                for s in summaries
                if scenario in s.get("by_scenario", {})
                and s["by_scenario"][scenario].get("reward") is not None
            ]
            if not rl_vals:
                continue
            k = len(rl_vals)
            mean = float(np.mean(rl_vals))
            std = float(np.std(rl_vals, ddof=1)) if k > 1 else 0.0
            se = std / (k**0.5) if k else 0.0

            best_name, best, oracle = None, None, None
            for name, entry in baselines.items():
                reward = entry.get("by_scenario", {}).get(scenario, {}).get("reward")
                if reward is None:
                    continue
                if name in ORACLE_BASELINES:
                    oracle = reward if oracle is None else max(oracle, reward)
                elif best is None or reward > best:
                    best_name, best = name, reward

            delta = None if best is None else mean - best
            half_ci = t95(k) * se
            win = None
            if delta is not None and se > 0:
                win = bool(abs(delta) > half_ci and delta > 0)
            per_scenario[scenario] = {
                "n_seeds": k,
                "rl_mean": mean,
                "rl_std": std,
                "rl_se": se,
                "best_baseline": best_name,
                "best": best,
                "delta": delta,
                "half_ci": half_ci,
                "win": win,
                "oracle": oracle,
                "oracle_gap": None if oracle is None else mean - oracle,
            }
        result[config] = per_scenario
    return result


def config_scores(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-config balanced_score across seeds (mean/std), for the headline number."""
    import numpy as np

    from src.train_acac import checkpoint_score

    by_config: dict[str, list[float]] = {}
    for run in runs:
        score = checkpoint_score(run["summary"])
        if score is not None:
            by_config.setdefault(run["config"], []).append(score)
    out: dict[str, dict[str, float]] = {}
    for config, scores in by_config.items():
        k = len(scores)
        out[config] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if k > 1 else 0.0,
            "n_seeds": k,
        }
    return out


def compare_configs(
    runs: list[dict[str, Any]], config_a: str, config_b: str
) -> dict[str, dict[str, Any]]:
    """Per-scenario Δ = mean(config_b) - mean(config_a) across seeds, with CI.

    ``resolved`` is True when |Δ| exceeds the combined across-seed 95% interval
    (i.e. config_b differs from config_a beyond seed noise).
    """
    import numpy as np

    def rl_by_scenario(config: str) -> dict[str, list[float]]:
        vals: dict[str, list[float]] = {}
        for run in runs:
            if run["config"] != config:
                continue
            for scenario, row in run["summary"].get("by_scenario", {}).items():
                if row.get("reward") is not None:
                    vals.setdefault(scenario, []).append(row["reward"])
        return vals

    a, b = rl_by_scenario(config_a), rl_by_scenario(config_b)
    out: dict[str, dict[str, Any]] = {}
    for scenario in sorted(set(a) & set(b)):
        va, vb = a[scenario], b[scenario]
        mean_a, mean_b = float(np.mean(va)), float(np.mean(vb))
        se_a = (float(np.std(va, ddof=1)) / len(va) ** 0.5) if len(va) > 1 else 0.0
        se_b = (float(np.std(vb, ddof=1)) / len(vb) ** 0.5) if len(vb) > 1 else 0.0
        se = (se_a**2 + se_b**2) ** 0.5
        delta = mean_b - mean_a
        half_ci = t95(min(len(va), len(vb))) * se
        out[scenario] = {
            "delta": delta,
            "half_ci": half_ci,
            "resolved": bool(abs(delta) > half_ci) if se > 0 else None,
        }
    return out


def format_multiseed_report(
    runs: list[dict[str, Any]],
    *,
    seeds: list[int],
) -> str:
    """Render the aggregated multi-seed analysis as copy-paste-friendly text."""
    agg = aggregate_runs(runs)
    scores = config_scores(runs)
    lines = [f"=== multi-seed summary | seeds={seeds} ==="]
    lines.append("(higher reward = better; 0=best realistic baseline; SJF* = clairvoyant oracle ceiling)")
    for config in agg:
        sc = scores.get(config, {})
        score_str = (
            f"{sc['mean']:+.3f} ± {sc['std']:.3f} (n={sc['n_seeds']} seeds)" if sc else "n/a"
        )
        lines.append("")
        lines.append(f"[{config}]  balanced_score = {score_str}")
        lines.append(
            f"  {'scenario':<13}{'rl (mean±std)':>20}   vs best realistic"
            f"            | SJF* oracle gap"
        )
        for scenario in sorted(agg[config]):
            s = agg[config][scenario]
            verdict = "?   " if s["win"] is None else ("WIN " if s["win"] else "tie ")
            best = "-" if s["best"] is None else f"{s['best_baseline']} {s['best']:+.1f}"
            delta = "-" if s["delta"] is None else f"Δ {s['delta']:+.1f} ±{s['half_ci']:.1f}"
            oracle = "-" if s["oracle_gap"] is None else f"{s['oracle_gap']:+.1f}"
            lines.append(
                f"  {scenario:<13}{s['rl_mean']:>10.1f} ±{s['rl_std']:6.1f}   "
                f"vs {best:<16} {delta:<16} {verdict}| {oracle}"
            )

    configs = list(agg)
    if len(configs) == 2:
        cmp = compare_configs(runs, configs[0], configs[1])
        lines.append("")
        lines.append(f"=== {configs[1]} vs {configs[0]} (Δ = {configs[1]} - {configs[0]}, across seeds) ===")
        for scenario in sorted(cmp):
            c = cmp[scenario]
            tag = "?" if c["resolved"] is None else ("resolved" if c["resolved"] else "within seed noise")
            lines.append(f"  {scenario:<13} Δ {c['delta']:+8.1f} ± {c['half_ci']:6.1f}  ({tag})")
    return "\n".join(lines)


# --- Orchestration (impure: spawns trainings) ------------------------------

_RECIPE = [
    "--reward-mode", "latency_flow",
    "--lambda-flow", "1.0", "--response-weight", "1.5", "--lambda-energy", "0",
    "--lambda-context-switch", "3",
    "--arrival-rate", "1.0", "--episode-time", "40", "--max-tasks", "32",
    "--hidden-dim", "128", "--rollout-episodes", "16",
    "--clip-ratio", "0.2", "--entropy-coef", "0.01", "--entropy-coef-final", "0.005",
    "--update-epochs", "4", "--lr-anneal-final-frac", "0.1",
    "--num-minibatches", "4", "--replay-capacity", "0",
    "--advantage-norm", "per_scenario",
]


def build_train_command(
    *,
    seed: int,
    output_dir: Path,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    workers: int,
    device: str,
    warm: bool,
    pretrained: Path,
) -> list[str]:
    """Standard `# 300` recipe (mirrors the notebook) + per-run overrides.

    Built-in held-out test is disabled (``--test-episodes 0``); the final number
    comes from eval_checkpoint at a chosen sample instead.
    """
    cmd = [sys.executable, "-m", "src.train_acac", *_RECIPE]
    cmd += [
        "--episodes", str(episodes),
        "--eval-every", str(eval_every),
        "--eval-episodes", str(eval_episodes),
        "--test-episodes", "0",
        "--rollout-workers", str(workers),
        "--device", device,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    if warm:
        cmd += ["--pretrained-actors", str(pretrained)]
    return cmd


def _parse_seeds(spec: str) -> list[int]:
    return [int(s) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed train + held-out eval for ACAC.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma list, e.g. 0,1,2.")
    parser.add_argument("--configs", type=str, default="scratch,warm", help="scratch and/or warm.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=40, help="Train-time eval episodes.")
    parser.add_argument(
        "--eval-scenario-episodes",
        type=str,
        default="balanced=80,bg_heavy=80,burst_stress=200,ui_heavy=80",
        help="Held-out eval episodes per scenario for the final number.",
    )
    parser.add_argument("--eval-seed", type=int, default=30000)
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("outputs/sjf_imitation/actors.pt"),
        help="SJF-imitation actors for warm start.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true", help="Retrain even if best.pt exists.")
    parser.add_argument("--skip-train", action="store_true", help="Aggregate existing runs only.")
    args = parser.parse_args()

    from src.eval_checkpoint import evaluate_checkpoint, parse_scenario_episodes
    from src.train_acac import parse_workload_scenarios

    seeds = _parse_seeds(args.seeds)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    scenarios = parse_workload_scenarios(args.scenarios)
    scenario_episodes = parse_scenario_episodes(
        args.eval_scenario_episodes, default=args.eval_episodes, scenarios=scenarios
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if "warm" in configs and not args.pretrained.exists() and not args.skip_train:
        raise SystemExit(
            f"warm start needs {args.pretrained} (run `python -m src.train_sjf_imitation "
            f"--device {args.device} --output {args.pretrained}` first), or drop 'warm' from --configs."
        )

    runs: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            output_dir = args.out / f"{config}_seed{seed}"
            best = output_dir / "best.pt"
            if args.force or (not best.exists() and not args.skip_train):
                cmd = build_train_command(
                    seed=seed,
                    output_dir=output_dir,
                    episodes=args.episodes,
                    eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes,
                    workers=args.workers,
                    device=args.device,
                    warm=(config == "warm"),
                    pretrained=args.pretrained,
                )
                print(f"\n>>> training {config} seed={seed} -> {output_dir}", flush=True)
                subprocess.run(cmd, check=True)
            if not best.exists():
                print(f"[warn] no best.pt for {config} seed={seed}, skipping", flush=True)
                continue
            print(f">>> eval {config} seed={seed} ({best})", flush=True)
            summary, meta = evaluate_checkpoint(
                best,
                scenario_episodes=scenario_episodes,
                base_seed=args.eval_seed,
                device=args.device,
            )
            runs.append({"config": config, "seed": seed, "summary": summary, "meta": meta})

    if not runs:
        raise SystemExit("No runs to aggregate (train first, or check --out).")

    report = format_multiseed_report(runs, seeds=seeds)
    print("\n" + report)
    out_json = args.out / "multiseed_summary.json"
    out_json.write_text(
        json.dumps({"seeds": seeds, "runs": runs, "aggregate": aggregate_runs(runs)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
