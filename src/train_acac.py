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
from src.train_logging import (
    configure_logging,
    get_logger,
    log_episode_eval,
    log_episode_train,
)


CHECKPOINT_VERSION = "learnable_actor_filter_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-config ACAC sanity training.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--eval-scenarios",
        type=str,
        default="all",
        help=(
            "Comma-separated workload scenarios for fixed validation eval, "
            "or 'all'. Uses held-out eval seeds."
        ),
    )
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20_000)
    parser.add_argument(
        "--test-scenarios",
        type=str,
        default="all",
        help=(
            "Comma-separated workload scenarios for final held-out test eval, "
            "or 'all'. Uses seeds separate from train/eval."
        ),
    )
    parser.add_argument("--rollout-episodes", type=int, default=16)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help=(
            "Parallel rollout worker processes. 1 (default) runs in-process. "
            ">1 collects episodes across CPU cores (workers do CPU inference; "
            "the GPU policy is used only for the update)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--episode-time", type=float, default=80.0)
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument(
        "--train-scenarios",
        type=str,
        default="all",
        help=(
            "Comma-separated workload scenarios sampled round-robin during "
            "training rollout, or 'all'. Choices: balanced, ui_heavy, "
            "bg_heavy, burst_stress."
        ),
    )
    parser.add_argument(
        "--disable-preemption",
        action="store_true",
        help="Run the non-preemptive environment (ablation).",
    )
    parser.add_argument("--progress-work", type=float, default=0.0)
    parser.add_argument("--completion", type=float, default=0.0)
    parser.add_argument("--completion-work", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.1)
    parser.add_argument("--lambda-starvation", type=float, default=0.05)
    parser.add_argument("--lambda-latency", type=float, default=0.5)
    parser.add_argument("--starvation-max-wait-weight", type=float, default=0.5)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="event_shaped",
        choices=["event_shaped", "event_cost", "completion_only", "latency_flow"],
        help=(
            "Reward formulation. 'latency_flow' = priority-weighted flow-time "
            "(dense per-step turnaround penalty; idle is never free). Recommended "
            "for a latency objective."
        ),
    )
    parser.add_argument(
        "--lambda-flow",
        type=float,
        default=1.0,
        help="LATENCY_FLOW: priority-weighted flow-time penalty weight.",
    )
    parser.add_argument(
        "--lambda-context-switch",
        type=float,
        default=1.0,
        help="Context-switch (preemption) penalty weight. Raise to discourage thrashing.",
    )
    parser.add_argument(
        "--response-weight",
        type=float,
        default=1.5,
        help="LATENCY_FLOW: extra weight while a task still waits for its first run.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--actor-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto' (CUDA if available else CPU), 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--pretrained-actors",
        type=Path,
        default=None,
        help="Optional SJF imitation checkpoint used to initialize actor weights.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/acac_p2e2"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from, such as outputs/acac_p2e2/latest.pt.",
    )
    args = parser.parse_args()
    args.enable_preemption = not args.disable_preemption
    args.train_scenarios = parse_workload_scenarios(args.train_scenarios)
    args.eval_scenarios = parse_workload_scenarios(args.eval_scenarios)
    args.test_scenarios = parse_workload_scenarios(args.test_scenarios)

    logger = configure_logging(args.output_dir)

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

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda"):
        # TF32 matmul/conv: large speedup on Ampere+ GPUs (Colab T4/A100) at
        # negligible precision cost for these small MLP/attention heads.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("using GPU device=%s (TF32 enabled)", args.device)

    config = ACACConfig(
        hidden_dim=args.hidden_dim,
        allow_noop=True,
        reward_scale=args.reward_scale,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
    )
    policy = TorchACACPolicy(config, device=args.device)
    if args.pretrained_actors is not None:
        checkpoint = torch.load(
            args.pretrained_actors,
            map_location=policy.device,
            weights_only=False,
        )
        policy.actors.load_state_dict(checkpoint["actors_state_dict"])
        logger.info("loaded pretrained actors=%s", args.pretrained_actors)
    trainer = ACACTrainer(policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    start_episode = 1
    best_eval_reward = float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=policy.device, weights_only=False)
        validate_checkpoint_version(checkpoint)
        policy.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint["episode"]) + 1
        best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
        logger.info(
            "resumed checkpoint=%s next_episode=%d", args.resume, start_episode
        )

    logger.info("=== ACAC single-config sanity training ===")
    logger.info("config: %s", asdict(config))
    logger.info("args: %s", vars(args))
    logger.info(
        "logs=%s latest=%s best=%s", metrics_path, latest_path, best_path
    )

    executor = None
    if args.rollout_workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=args.rollout_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_rollout_worker,
            initargs=(config, rollout_env_kwargs(args)),
        )
        logger.info(
            "parallel rollout: workers=%d (spawn, CPU inference)",
            args.rollout_workers,
        )

    try:
        run_training_loop(
            policy=policy,
            trainer=trainer,
            args=args,
            config=config,
            logger=logger,
            torch=torch,
            executor=executor,
            metrics_path=metrics_path,
            latest_path=latest_path,
            best_path=best_path,
            start_episode=start_episode,
            best_eval_reward=best_eval_reward,
        )
        if args.test_episodes > 0:
            if best_path.exists():
                checkpoint = torch.load(
                    best_path,
                    map_location=policy.device,
                    weights_only=False,
                )
                policy.load_state_dict(checkpoint["model_state_dict"])
                logger.info("loaded best checkpoint for held-out test=%s", best_path)
            test_summary = evaluate_policy(
                policy,
                args,
                base_seed=args.test_seed,
                episodes=args.test_episodes,
                scenarios=args.test_scenarios,
            )
            append_jsonl(
                metrics_path,
                {
                    "episode": args.episodes,
                    "split": "test",
                    "test": test_summary,
                },
            )
            logger.info("=== held-out test ===")
            log_episode_eval(test_summary)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def run_training_loop(
    *,
    policy,
    trainer,
    args: argparse.Namespace,
    config,
    logger,
    torch,
    executor,
    metrics_path: Path,
    latest_path: Path,
    best_path: Path,
    start_episode: int,
    best_eval_reward: float,
) -> None:
    for episode_idx in range(start_episode, args.episodes + 1):
        rollout, metrics = collect_training_rollout(
            policy,
            args,
            gamma=config.gamma,
            episode_idx=episode_idx,
            executor=executor,
        )
        if len(rollout) == 0:
            logger.warning("ep %d skipped empty rollout", episode_idx)
            continue

        stats = trainer.update(rollout)
        total_reward = rollout.total_env_reward / args.rollout_episodes
        should_eval = episode_idx == start_episode or episode_idx % args.eval_every == 0
        eval_summary = (
            evaluate_policy(
                policy,
                args,
                base_seed=args.eval_seed,
                episodes=args.eval_episodes,
                scenarios=args.eval_scenarios,
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
            train_scenarios=training_scenario_counts(args, episode_idx),
        )
        append_jsonl(metrics_path, log_row)

        log_episode_train(
            episode_idx=episode_idx,
            total_reward=total_reward,
            rollout=rollout,
            metrics=metrics,
            stats=stats,
        )

        if eval_summary is not None:
            log_episode_eval(eval_summary)
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


def rollout_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Environment constructor kwargs (everything except the per-episode seed).

    Shared by the in-process ``make_env`` and the rollout workers so a worker's
    environment is byte-for-byte identical to the sequential one for a given
    seed. The workload scenario is supplied per rollout episode so training can
    round-robin across scenarios while evaluation can stay fixed.
    """
    return {
        "core_config": {CoreType.P: 2, CoreType.E: 2},
        "arrival_rate": args.arrival_rate,
        "episode_time": args.episode_time,
        "max_tasks": args.max_tasks,
        "reward_weights": reward_weights_from_args(args),
        "enable_preemption": getattr(args, "enable_preemption", False),
        "reward_mode": getattr(args, "reward_mode", "event_shaped"),
    }


def make_env(
    args: argparse.Namespace,
    *,
    seed: int,
    workload_scenario: WorkloadScenario | str = WorkloadScenario.BALANCED,
) -> SchedulerEnv:
    return SchedulerEnv(
        seed=seed,
        workload_scenario=WorkloadScenario(workload_scenario),
        **rollout_env_kwargs(args),
    )


# --- Parallel rollout ------------------------------------------------------
# The environment is a pure-Python discrete-event simulation (CPU bound), so
# the rollout cannot run on the GPU. To use the host's cores we run several
# independent episodes in worker processes. Workers do CPU-only inference with
# a snapshot of the current weights; the main process keeps the GPU policy for
# the gradient update. We use the 'spawn' start method so children never
# inherit the parent's CUDA context (forking a CUDA process is unsafe).

_ROLLOUT_WORKER: dict[str, Any] = {}


def _init_rollout_worker(config: Any, env_kwargs: dict[str, Any]) -> None:
    import torch

    from src.rl.trainer import TorchACACPolicy

    # One torch thread per worker avoids oversubscribing the (few) Colab cores.
    torch.set_num_threads(1)
    _ROLLOUT_WORKER["policy"] = TorchACACPolicy(config, device="cpu")
    _ROLLOUT_WORKER["env_kwargs"] = env_kwargs


def _run_rollout_worker(
    task: tuple[dict[str, Any], int, float, WorkloadScenario]
) -> tuple[RolloutBuffer, Any]:
    import torch

    state_dict, seed, gamma, scenario = task
    policy = _ROLLOUT_WORKER["policy"]
    policy.load_state_dict(state_dict)
    policy.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        env = SchedulerEnv(
            seed=seed,
            workload_scenario=scenario,
            **_ROLLOUT_WORKER["env_kwargs"],
        )
        buffer = collect_episode(env, policy, seed=seed, gamma=gamma)
    return buffer, env.metrics()


def rollout_seeds(args: argparse.Namespace, episode_idx: int) -> list[int]:
    return [
        args.seed + (episode_idx - 1) * args.rollout_episodes + offset + 1
        for offset in range(args.rollout_episodes)
    ]


def training_scenarios(args: argparse.Namespace) -> tuple[WorkloadScenario, ...]:
    return parse_workload_scenarios(getattr(args, "train_scenarios", (WorkloadScenario.BALANCED,)))


def parse_workload_scenarios(
    value: str | WorkloadScenario | list[str | WorkloadScenario] | tuple[str | WorkloadScenario, ...],
) -> tuple[WorkloadScenario, ...]:
    if isinstance(value, WorkloadScenario):
        return (value,)
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",") if item.strip()]
        if len(raw_items) == 1 and raw_items[0].lower() == "all":
            return tuple(WorkloadScenario)
    else:
        raw_items = list(value)

    scenarios = tuple(WorkloadScenario(item) for item in raw_items)
    if not scenarios:
        raise ValueError("At least one training scenario must be provided.")
    return scenarios


def rollout_scenarios(args: argparse.Namespace, episode_idx: int) -> list[WorkloadScenario]:
    scenarios = training_scenarios(args)
    start = (episode_idx - 1) * args.rollout_episodes
    return [
        scenarios[(start + offset) % len(scenarios)]
        for offset in range(args.rollout_episodes)
    ]


def training_scenario_counts(
    args: argparse.Namespace,
    episode_idx: int,
) -> dict[str, int]:
    counts = {scenario.value: 0 for scenario in training_scenarios(args)}
    for scenario in rollout_scenarios(args, episode_idx):
        counts[scenario.value] += 1
    return counts


def collect_training_rollout(
    policy,
    args: argparse.Namespace,
    *,
    gamma: float,
    episode_idx: int,
    executor,
) -> tuple[RolloutBuffer, Any]:
    """Collect ``rollout_episodes`` episodes and merge them in seed order.

    Merging in seed order (not completion order) keeps episode ids and joint
    indices identical to sequential collection, so credit assignment is
    unchanged regardless of which worker finished first. Returns the merged
    buffer and the metrics of the last-seed episode (matching the sequential
    ``env.metrics()`` used for logging).
    """

    seeds = rollout_seeds(args, episode_idx)
    scenarios = rollout_scenarios(args, episode_idx)
    rollout = RolloutBuffer()
    last_metrics = None

    if executor is None:
        for seed, scenario in zip(seeds, scenarios, strict=True):
            env = make_env(args, seed=seed, workload_scenario=scenario)
            rollout.extend(collect_episode(env, policy, seed=seed, gamma=gamma))
            last_metrics = env.metrics()
        return rollout, last_metrics

    cpu_state = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    tasks = [
        (cpu_state, seed, gamma, scenario)
        for seed, scenario in zip(seeds, scenarios, strict=True)
    ]
    for buffer, metrics in executor.map(_run_rollout_worker, tasks):
        rollout.extend(buffer)
        last_metrics = metrics
    return rollout, last_metrics


def reward_weights_from_args(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        progress_work=getattr(args, "progress_work", 0.0),
        completion=getattr(args, "completion", 0.0),
        completion_work=getattr(args, "completion_work", 0.0),
        energy=getattr(args, "lambda_energy", 0.1),
        starvation=getattr(args, "lambda_starvation", 0.05),
        latency=getattr(args, "lambda_latency", 0.5),
        starvation_max_wait_weight=getattr(args, "starvation_max_wait_weight", 0.5),
        flow_time=getattr(args, "lambda_flow", 1.0),
        context_switch=getattr(args, "lambda_context_switch", 1.0),
        response_weight=getattr(args, "response_weight", 1.5),
    )


def evaluate_policy(
    policy,
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int = 5,
    scenarios: tuple[WorkloadScenario, ...] | None = None,
) -> dict[str, Any]:
    scenarios = scenarios or (WorkloadScenario.BALANCED,)
    summary = evaluate_rl_policy(
        policy,
        args,
        base_seed=base_seed,
        episodes=episodes,
        deterministic=True,
        scenarios=scenarios,
    )
    summary["sampled"] = evaluate_rl_policy(
        policy,
        args,
        base_seed=base_seed,
        episodes=episodes,
        deterministic=False,
        scenarios=scenarios,
    )
    summary["baselines"] = cached_evaluate_baselines(
        args, base_seed=base_seed, episodes=episodes, scenarios=scenarios
    )
    return summary


# Baselines do not depend on the learned policy and the evaluation seed/config
# is fixed across a training run, so their summary is identical at every eval
# point. Computing it once instead of every `--eval-every` removes the bulk of
# the per-eval episode cost (3 baselines x eval-episodes each).
_BASELINE_CACHE: dict[tuple[Any, ...], dict[str, dict[str, float]]] = {}


def cached_evaluate_baselines(
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int,
    scenarios: tuple[WorkloadScenario, ...],
) -> dict[str, dict[str, float]]:
    key = (
        base_seed,
        episodes,
        tuple(scenario.value for scenario in scenarios),
        getattr(args, "arrival_rate", 1.0),
        getattr(args, "episode_time", 80.0),
        getattr(args, "max_tasks", 64),
        getattr(args, "enable_preemption", False),
        getattr(args, "progress_work", 0.0),
        getattr(args, "completion", 0.0),
        getattr(args, "completion_work", 0.0),
        getattr(args, "lambda_energy", 0.1),
        getattr(args, "lambda_starvation", 0.05),
        getattr(args, "lambda_latency", 0.5),
        getattr(args, "starvation_max_wait_weight", 0.5),
        getattr(args, "reward_mode", "event_shaped"),
        getattr(args, "lambda_flow", 1.0),
        getattr(args, "lambda_context_switch", 1.0),
        getattr(args, "response_weight", 1.5),
    )
    cached = _BASELINE_CACHE.get(key)
    if cached is None:
        cached = evaluate_baselines(
            args,
            base_seed=base_seed,
            episodes=episodes,
            scenarios=scenarios,
        )
        _BASELINE_CACHE[key] = cached
    return cached


def evaluate_rl_policy(
    policy,
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int,
    deterministic: bool,
    scenarios: tuple[WorkloadScenario, ...],
) -> dict[str, Any]:
    rewards = []
    completed = []
    throughputs = []
    diagnostics = []
    action_summaries = []
    preemptions = []
    scenario_labels = []
    with preserve_torch_rng(seed=base_seed, enabled=not deterministic):
        for offset in range(episodes):
            scenario = scenarios[offset % len(scenarios)]
            scenario_labels.append(scenario.value)
            env = make_env(
                args,
                seed=base_seed + offset,
                workload_scenario=scenario,
            )
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
            preemptions.append(rollout.preemptions)

    return {
        "reward": float(np.mean(rewards)),
        "completed": float(np.mean(completed)),
        "throughput": float(np.mean(throughputs)),
        "preemptions": float(np.mean(preemptions)),
        "scenario_counts": count_labels(scenario_labels),
        "reward_diagnostics": mean_dict(diagnostics),
        "actions": mean_dict(action_summaries),
    }


def evaluate_baselines(
    args: argparse.Namespace,
    *,
    base_seed: int,
    episodes: int,
    scenarios: tuple[WorkloadScenario, ...],
) -> dict[str, dict[str, float]]:
    from src.baselines import EASLikePolicy, MLFQPolicy, RandomPolicy, SJFLikePolicy, run_episode

    policies = [RandomPolicy(seed=base_seed), MLFQPolicy(), SJFLikePolicy(), EASLikePolicy()]
    summaries = {}
    for baseline in policies:
        results = [
            run_episode(
                make_env(
                    args,
                    seed=base_seed + offset,
                    workload_scenario=scenarios[offset % len(scenarios)],
                ),
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


def count_labels(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


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
    # Selected-task features only apply to dispatch/preempt actions; NO-OP
    # (action 0) selects no task and is excluded from the per-task averages.
    dispatched = [transition for transition in rollout.transitions if transition.action > 0]
    if dispatched:
        selected_tasks = np.stack(
            [
                transition.obs.ready_queue[
                    transition.agent_index,
                    transition.action - 1,
                ]
                for transition in dispatched
            ]
        )
    else:
        selected_tasks = np.zeros((1, rollout.transitions[0].obs.ready_queue.shape[-1]))
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
    train_scenarios: dict[str, int] | None = None,
) -> dict[str, Any]:
    elapsed_times = [transition.elapsed_time for transition in rollout.transitions]
    return {
        "episode": episode_idx,
        "transitions": len(rollout),
        "joint_intervals": len(rollout.joint_transitions),
        "env_steps": rollout.env_steps,
        "conflicts": rollout.conflicts,
        "invalid_actions": rollout.invalid_actions,
        "preemptions": rollout.preemptions,
        "decisions": rollout.decisions,
        "mean_task_choices": rollout.mean_task_choices,
        "max_task_choices": rollout.max_task_choices,
        "forced_decision_fraction": rollout.forced_decision_fraction,
        "reward": total_reward,
        "train_scenarios": train_scenarios or {},
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
        "checkpoint_version": CHECKPOINT_VERSION,
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
    get_logger().info("saved checkpoint=%s episode=%d", path, episode_idx)


def validate_checkpoint_version(checkpoint: dict[str, Any]) -> None:
    checkpoint_version = checkpoint.get("checkpoint_version")
    if checkpoint_version != CHECKPOINT_VERSION:
        raise SystemExit(
            "Checkpoint version mismatch: "
            f"expected {CHECKPOINT_VERSION!r}, got {checkpoint_version!r}. "
            "Start a fresh output directory after changing the training algorithm."
        )


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: serialize_arg_value(value) for key, value in vars(args).items()}


def serialize_arg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, WorkloadScenario):
        return value.value
    if isinstance(value, tuple):
        return [serialize_arg_value(item) for item in value]
    if isinstance(value, list):
        return [serialize_arg_value(item) for item in value]
    return value


class EvaluationPolicy:
    def __init__(self, policy, *, deterministic: bool) -> None:
        self.policy = policy
        self.deterministic = deterministic

    def act(self, batch):
        return self.policy.act(batch, deterministic=self.deterministic)


if __name__ == "__main__":
    main()
