from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import collect_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-config ACAC sanity training.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0.5)
    parser.add_argument("--episode-time", type=float, default=80.0)
    parser.add_argument("--max-tasks", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/acac_p2e2"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from, such as outputs/acac_p2e2/latest.pt.",
    )
    args = parser.parse_args()

    try:
        import torch

        from src.rl.trainer import ACACConfig, ACACTrainer, TorchACACPolicy
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for training. Run this script in Colab or "
                "install torch in the active environment."
            ) from exc
        raise

    config = ACACConfig(hidden_dim=args.hidden_dim, allow_noop=False)
    policy = TorchACACPolicy(config, device=args.device)
    trainer = ACACTrainer(policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    start_episode = 1
    best_eval_reward = float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=policy.device, weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint["episode"]) + 1
        best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
        print(f"resumed checkpoint={args.resume} next_episode={start_episode}")

    print("ACAC single-config sanity training")
    print({"config": asdict(config), "args": vars(args)})
    print(f"logs={metrics_path} latest={latest_path} best={best_path}")

    for episode_idx in range(start_episode, args.episodes + 1):
        env = make_env(args, seed=args.seed + episode_idx)
        rollout = collect_episode(env, policy, seed=args.seed + episode_idx)
        if len(rollout) == 0:
            print(f"episode={episode_idx} skipped empty rollout")
            continue

        stats = trainer.update(rollout)
        total_reward = sum(transition.reward for transition in rollout.transitions)
        metrics = env.metrics()
        should_eval = episode_idx == start_episode or episode_idx % args.eval_every == 0
        eval_summary = (
            evaluate_policy(
                policy,
                args,
                base_seed=args.eval_seed,
                episodes=args.eval_episodes,
            )
            if should_eval
            else None
        )
        log_row = build_log_row(
            episode_idx=episode_idx,
            rollout=rollout,
            total_reward=total_reward,
            metrics=metrics,
            stats=stats,
            eval_summary=eval_summary,
        )
        append_jsonl(metrics_path, log_row)

        if eval_summary is not None:
            print(
                "episode={episode} transitions={transitions} reward={reward:.3f} "
                "completed={completed}/{total} throughput={throughput:.3f} "
                "turnaround={turnaround} loss={loss:.3f} value_loss={value_loss:.3f} "
                "entropy={entropy:.3f} kl={kl:.4f} clip_frac={clip_fraction:.3f} "
                "conflicts={conflicts} eval_reward={eval_reward:.3f} "
                "eval_completed={eval_completed:.1f} eas_reward={eas_reward:.3f}".format(
                    episode=episode_idx,
                    transitions=len(rollout),
                    reward=total_reward,
                    completed=metrics.completed_tasks,
                    total=metrics.total_tasks,
                    throughput=metrics.throughput,
                    turnaround=_fmt(metrics.mean_turnaround_time),
                    loss=stats.loss,
                    value_loss=stats.value_loss,
                    entropy=stats.entropy,
                    kl=stats.approx_kl,
                    clip_fraction=stats.clip_fraction,
                    conflicts=rollout.conflicts,
                    eval_reward=eval_summary["reward"],
                    eval_completed=eval_summary["completed"],
                    eas_reward=eval_summary["baselines"]["eas_like"]["reward"],
                )
            )
            if eval_summary["reward"] > best_eval_reward:
                best_eval_reward = eval_summary["reward"]
                save_checkpoint(
                    torch=torch,
                    path=best_path,
                    episode_idx=episode_idx,
                    policy=policy,
                    trainer=trainer,
                    config=config,
                    args=args,
                    best_eval_reward=best_eval_reward,
                    eval_summary=eval_summary,
                )

        if episode_idx % args.save_every == 0 or episode_idx == args.episodes:
            save_checkpoint(
                torch=torch,
                path=latest_path,
                episode_idx=episode_idx,
                policy=policy,
                trainer=trainer,
                config=config,
                args=args,
                best_eval_reward=best_eval_reward,
                eval_summary=eval_summary,
            )


def make_env(args: argparse.Namespace, *, seed: int) -> SchedulerEnv:
    return SchedulerEnv(
        core_config={CoreType.P: 2, CoreType.E: 2},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=args.arrival_rate,
        episode_time=args.episode_time,
        max_tasks=args.max_tasks,
        seed=seed,
    )


def evaluate_policy(
    policy,
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int = 5,
) -> dict[str, Any]:
    rewards = []
    completed = []
    throughputs = []
    for offset in range(episodes):
        env = make_env(args, seed=base_seed + offset)
        rollout = collect_episode(
            env,
            DeterministicPolicy(policy),
            seed=base_seed + offset,
        )
        rewards.append(sum(transition.reward for transition in rollout.transitions))
        metrics = env.metrics()
        completed.append(metrics.completed_tasks)
        throughputs.append(metrics.throughput)

    return {
        "reward": float(np.mean(rewards)),
        "completed": float(np.mean(completed)),
        "throughput": float(np.mean(throughputs)),
        "baselines": evaluate_baselines(args, base_seed=base_seed, episodes=episodes),
    }


def evaluate_baselines(
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int,
) -> dict[str, dict[str, float]]:
    from src.baselines import EASLikePolicy, RandomPolicy, SJFLikePolicy, run_episode

    policies = [RandomPolicy(seed=base_seed), SJFLikePolicy(), EASLikePolicy()]
    summaries = {}
    for baseline in policies:
        results = [
            run_episode(
                make_env(args, seed=base_seed + offset),
                baseline,
                seed=base_seed + offset,
            )
            for offset in range(episodes)
        ]
        summaries[baseline.name] = {
            "reward": float(np.mean([result.total_reward for result in results])),
            "completed": float(np.mean([result.metrics.completed_tasks for result in results])),
            "throughput": float(np.mean([result.metrics.throughput for result in results])),
        }
    return summaries


def build_log_row(
    *,
    episode_idx: int,
    rollout,
    total_reward: float,
    metrics,
    stats,
    eval_summary: dict[str, float] | None,
) -> dict[str, Any]:
    elapsed_times = [transition.elapsed_time for transition in rollout.transitions]
    return {
        "episode": episode_idx,
        "transitions": len(rollout),
        "env_steps": rollout.env_steps,
        "conflicts": rollout.conflicts,
        "invalid_actions": rollout.invalid_actions,
        "reward": total_reward,
        "mean_elapsed_time": float(np.mean(elapsed_times)),
        "metrics": metrics.as_dict(),
        "update": asdict(stats),
        "evaluation": eval_summary,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def save_checkpoint(
    *,
    torch,
    path: Path,
    episode_idx: int,
    policy,
    trainer,
    config,
    args: argparse.Namespace,
    best_eval_reward: float,
    eval_summary: dict[str, float] | None,
) -> None:
    checkpoint = {
        "episode": episode_idx,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": asdict(config),
        "args": serialize_args(args),
        "best_eval_reward": best_eval_reward,
        "eval_metrics": eval_summary,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)
    print(f"saved checkpoint={path} episode={episode_idx}")


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


class DeterministicPolicy:
    def __init__(self, policy) -> None:
        self.policy = policy

    def act(self, batch):
        return self.policy.act(batch, deterministic=True)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
