from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import collect_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-config ACAC sanity training.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0.5)
    parser.add_argument("--episode-time", type=float, default=80.0)
    parser.add_argument("--max-tasks", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    try:
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

    print("ACAC single-config sanity training")
    print({"config": asdict(config), "args": vars(args)})

    for episode_idx in range(1, args.episodes + 1):
        env = make_env(args, seed=args.seed + episode_idx)
        rollout = collect_episode(env, policy, seed=args.seed + episode_idx)
        if len(rollout) == 0:
            print(f"episode={episode_idx} skipped empty rollout")
            continue

        stats = trainer.update(rollout)
        total_reward = sum(transition.reward for transition in rollout.transitions)
        metrics = env.metrics()

        if episode_idx == 1 or episode_idx % args.eval_every == 0:
            eval_summary = evaluate_policy(policy, args, base_seed=args.seed + 10_000 + episode_idx)
            print(
                "episode={episode} transitions={transitions} reward={reward:.3f} "
                "completed={completed}/{total} throughput={throughput:.3f} "
                "turnaround={turnaround} loss={loss:.3f} value_loss={value_loss:.3f} "
                "entropy={entropy:.3f} eval_reward={eval_reward:.3f} "
                "eval_completed={eval_completed:.1f}".format(
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
                    eval_reward=eval_summary["reward"],
                    eval_completed=eval_summary["completed"],
                )
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


def evaluate_policy(policy, args: argparse.Namespace, *, base_seed: int, episodes: int = 3) -> dict[str, float]:
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
