from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.env import CoreType, RewardWeights, SchedulerEnv, WorkloadScenario
from src.rl import RolloutBuffer, collect_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-config ACAC sanity training.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--rollout-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--episode-time", type=float, default=80.0)
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument("--progress-work", type=float, default=0.0)
    parser.add_argument("--completion", type=float, default=0.0)
    parser.add_argument("--completion-work", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.1)
    parser.add_argument("--lambda-starvation", type=float, default=0.05)
    parser.add_argument("--lambda-latency", type=float, default=0.5)
    parser.add_argument("--starvation-max-wait-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--actor-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--update-epochs", type=int, default=2)
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

    config = ACACConfig(
        hidden_dim=args.hidden_dim,
        allow_noop=False,
        reward_scale=args.reward_scale,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
    )
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
        rollout = RolloutBuffer()
        env = None
        for rollout_offset in range(args.rollout_episodes):
            rollout_seed = (
                args.seed
                + (episode_idx - 1) * args.rollout_episodes
                + rollout_offset
                + 1
            )
            env = make_env(args, seed=rollout_seed)
            rollout.extend(
                collect_episode(
                    env,
                    policy,
                    seed=rollout_seed,
                    gamma=config.gamma,
                )
            )
        if len(rollout) == 0:
            print(f"episode={episode_idx} skipped empty rollout")
            continue

        assert env is not None
        stats = trainer.update(rollout)
        total_reward = rollout.total_env_reward / args.rollout_episodes
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
                "episode={episode} transitions={transitions} joints={joints} reward={reward:.3f} "
                "completed={completed}/{total} throughput={throughput:.3f} "
                "turnaround={turnaround} loss={loss:.3f} value_loss={value_loss:.3f} "
                "entropy={entropy:.3f} kl={kl:.4f} clip_frac={clip_fraction:.3f} "
                "actor_grad={actor_grad:.3f} critic_grad={critic_grad:.3f} "
                "adv_std={adv_std:.3f} "
                "conflicts={conflicts} choices={choices:.2f} forced={forced:.2f} "
                "eval_reward={eval_reward:.3f} sampled_eval_reward={sampled_eval_reward:.3f} "
                "eval_first={eval_first:.2f} sampled_first={sampled_first:.2f} "
                "eval_completed={eval_completed:.1f} random_reward={random_reward:.3f} "
                "sjf_reward={sjf_reward:.3f} eas_reward={eas_reward:.3f}".format(
                    episode=episode_idx,
                    transitions=len(rollout),
                    joints=len(rollout.joint_transitions),
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
                    actor_grad=stats.actor_grad_norm,
                    critic_grad=stats.critic_grad_norm,
                    adv_std=stats.advantage_std,
                    conflicts=rollout.conflicts,
                    choices=rollout.mean_task_choices,
                    forced=rollout.forced_decision_fraction,
                    eval_reward=eval_summary["reward"],
                    sampled_eval_reward=eval_summary["sampled"]["reward"],
                    eval_first=eval_summary["actions"]["first_slot_fraction"],
                    sampled_first=eval_summary["sampled"]["actions"]["first_slot_fraction"],
                    eval_completed=eval_summary["completed"],
                    random_reward=eval_summary["baselines"]["random"]["reward"],
                    sjf_reward=eval_summary["baselines"]["sjf_like"]["reward"],
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
        reward_weights=reward_weights_from_args(args),
    )


def reward_weights_from_args(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        progress_work=getattr(args, "progress_work", 0.0),
        completion=getattr(args, "completion", 0.0),
        completion_work=getattr(args, "completion_work", 0.0),
        energy=getattr(args, "lambda_energy", 0.1),
        starvation=getattr(args, "lambda_starvation", 0.05),
        latency=getattr(args, "lambda_latency", 0.5),
        starvation_max_wait_weight=getattr(args, "starvation_max_wait_weight", 0.5),
    )


def evaluate_policy(
    policy,
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int = 5,
) -> dict[str, Any]:
    summary = evaluate_rl_policy(
        policy,
        args,
        base_seed=base_seed,
        episodes=episodes,
        deterministic=True,
    )
    summary["sampled"] = evaluate_rl_policy(
        policy,
        args,
        base_seed=base_seed,
        episodes=episodes,
        deterministic=False,
    )
    summary["baselines"] = evaluate_baselines(args, base_seed=base_seed, episodes=episodes)
    return summary


def evaluate_rl_policy(
    policy,
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int,
    deterministic: bool,
) -> dict[str, Any]:
    rewards = []
    completed = []
    throughputs = []
    diagnostics = []
    action_summaries = []
    with preserve_torch_rng(seed=base_seed, enabled=not deterministic):
        for offset in range(episodes):
            env = make_env(args, seed=base_seed + offset)
            rollout = collect_episode(
                env,
                EvaluationPolicy(policy, deterministic=deterministic),
                seed=base_seed + offset,
                gamma=policy.config.gamma,
            )
            rewards.append(rollout.total_env_reward)
            metrics = env.metrics()
            completed.append(metrics.completed_tasks)
            throughputs.append(metrics.throughput)
            diagnostics.append(env.reward_diagnostics())
            action_summaries.append(summarize_rollout_actions(rollout))

    return {
        "reward": float(np.mean(rewards)),
        "completed": float(np.mean(completed)),
        "throughput": float(np.mean(throughputs)),
        "reward_diagnostics": mean_dict(diagnostics),
        "actions": mean_dict(action_summaries),
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
            "energy": float(np.mean([result.metrics.total_energy for result in results])),
            "turnaround": float(
                np.mean([result.metrics.mean_turnaround_time for result in results])
            ),
            "ready_wait": float(
                np.mean([result.metrics.mean_ready_wait_time for result in results])
            ),
            "reward_diagnostics": mean_dict(
                [result.reward_diagnostics for result in results]
            ),
        }
    return summaries


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def summarize_rollout_actions(rollout: RolloutBuffer) -> dict[str, float]:
    if not rollout.transitions:
        return {
            "mean_queue_slot": 0.0,
            "first_slot_fraction": 0.0,
            "mean_selected_wait": 0.0,
            "mean_selected_progress": 0.0,
            "mean_selected_latency": 0.0,
            "mean_selected_cpu_intensity": 0.0,
        }

    actions = np.asarray([transition.action for transition in rollout.transitions])
    selected_tasks = np.stack(
        [
            transition.obs.ready_queue[
                transition.agent_index,
                transition.action - 1,
            ]
            for transition in rollout.transitions
        ]
    )
    return {
        "mean_queue_slot": float(np.mean(actions)),
        "first_slot_fraction": float(np.mean(actions == 1)),
        "mean_selected_wait": float(np.mean(selected_tasks[:, 0])),
        "mean_selected_progress": float(np.mean(selected_tasks[:, 1])),
        "mean_selected_latency": float(np.mean(selected_tasks[:, 2])),
        "mean_selected_cpu_intensity": float(np.mean(selected_tasks[:, 3])),
    }


@contextmanager
def preserve_torch_rng(*, seed: int, enabled: bool):
    if not enabled:
        yield
        return

    import torch

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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
        "joint_intervals": len(rollout.joint_transitions),
        "env_steps": rollout.env_steps,
        "conflicts": rollout.conflicts,
        "invalid_actions": rollout.invalid_actions,
        "decisions": rollout.decisions,
        "mean_task_choices": rollout.mean_task_choices,
        "max_task_choices": rollout.max_task_choices,
        "forced_decision_fraction": rollout.forced_decision_fraction,
        "reward": total_reward,
        "transition_reward": float(
            sum(transition.reward for transition in rollout.transitions)
            / max(rollout.episodes, 1)
        ),
        "mean_elapsed_time": float(np.mean(elapsed_times)),
        "actions": summarize_rollout_actions(rollout),
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


class EvaluationPolicy:
    def __init__(self, policy, *, deterministic: bool) -> None:
        self.policy = policy
        self.deterministic = deterministic

    def act(self, batch):
        return self.policy.act(batch, deterministic=self.deterministic)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
