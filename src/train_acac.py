from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.env import CoreType, RewardWeights, SchedulerEnv, WorkloadScenario
from src.rl import ReplayBuffer, RolloutBuffer, collect_episode
from src.plot_metrics import ORACLE_BASELINES, summarize_scenario_significance
from src.train_logging import (
    configure_logging,
    get_logger,
    log_episode_eval,
    log_episode_train,
    log_scenario_significance,
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
    parser.add_argument(
        "--lr-anneal-final-frac",
        type=float,
        default=1.0,
        help=(
            "Linearly decay both learning rates to this fraction of their "
            "initial value over training. 1.0 (default) keeps them constant. "
            "Set <1 (e.g. 0.1) so update steps shrink as the policy converges — "
            "without it the advantage normalization keeps step size constant and "
            "the policy never settles (KL/clip stay flat, eval drifts)."
        ),
    )
    parser.add_argument(
        "--lr-anneal-episodes",
        type=int,
        default=None,
        help=(
            "Episodes over which the learning rate reaches its final fraction. "
            "None (default) uses the full --episodes budget."
        ),
    )
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Initial entropy bonus coefficient (start of the anneal schedule).",
    )
    parser.add_argument(
        "--entropy-coef-final",
        type=float,
        default=None,
        help=(
            "Final entropy coefficient to anneal toward. None (default) keeps "
            "--entropy-coef constant. Set lower than --entropy-coef to decay "
            "exploration as the policy converges (mitigates late policy drift)."
        ),
    )
    parser.add_argument(
        "--entropy-anneal-episodes",
        type=int,
        default=None,
        help=(
            "Episodes over which entropy-coef linearly reaches its final value. "
            "None (default) uses the full --episodes budget."
        ),
    )
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=1,
        help=(
            "Minibatches per update epoch over the macro-timeline intervals. "
            "1 (default) is a full-batch update."
        ),
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=0,
        help=(
            "Number of recent rollouts retained and reused each update "
            "(on-policy replay; PPO clipping bounds the staleness). 0 disables."
        ),
    )
    parser.add_argument(
        "--advantage-norm",
        type=str,
        default="global",
        choices=["global", "per_scenario", "per_episode"],
        help=(
            "Advantage normalization scope. 'global' (default) standardizes "
            "advantages across the whole merged rollout, so the "
            "largest-magnitude scenario (e.g. burst_stress) dominates the "
            "gradient. 'per_scenario' (recommended for multi-scenario training) "
            "standardizes within each workload scenario, equalizing scenarios of "
            "very different reward magnitude while keeping the genuine "
            "between-episode spread within a scenario. 'per_episode' standardizes "
            "every rollout episode on its own (also flattens within-scenario "
            "episode differences)."
        ),
    )
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

    # Eval/test split scenarios round-robin; a count that is not a multiple of the
    # scenario count samples some scenarios more than others (noisy per-scenario
    # numbers, which feed the best-checkpoint score). Warn so it is not silent.
    for split, episodes, scenarios in (
        ("eval", args.eval_episodes, args.eval_scenarios),
        ("test", args.test_episodes, args.test_scenarios),
    ):
        if episodes and len(scenarios) > 1 and episodes % len(scenarios) != 0:
            logger.warning(
                "%s-episodes=%d is not a multiple of %d scenarios -> uneven "
                "per-scenario sampling (%d-%d each). Use a multiple of %d.",
                split,
                episodes,
                len(scenarios),
                episodes // len(scenarios),
                -(-episodes // len(scenarios)),
                len(scenarios),
            )

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
        num_minibatches=args.num_minibatches,
        advantage_norm=args.advantage_norm,
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
    # Best-checkpoint selection score (scenario-balanced, scale-free — see
    # checkpoint_score). Persisted under the legacy "best_eval_reward" key.
    best_eval_score = float("-inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=policy.device, weights_only=False)
        validate_checkpoint_version(checkpoint)
        policy.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint["episode"]) + 1
        best_eval_score = float(checkpoint.get("best_eval_reward", float("-inf")))
        logger.info(
            "resumed checkpoint=%s next_iter=%d", args.resume, start_episode
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
            best_eval_score=best_eval_score,
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
            test_summary["balanced_score"] = checkpoint_score(test_summary)
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
            log_scenario_significance(summarize_scenario_significance(test_summary))
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
    best_eval_score: float,
) -> None:
    replay = ReplayBuffer(capacity=getattr(args, "replay_capacity", 0))
    # Base LRs come from config (not the optimizer) so a resumed run re-derives
    # the schedule from scratch instead of compounding a previously-scaled LR.
    lr_annealing = float(getattr(args, "lr_anneal_final_frac", 1.0)) != 1.0
    base_lrs = [config.actor_learning_rate, config.critic_learning_rate]
    for episode_idx in range(start_episode, args.episodes + 1):
        rollout, metrics = collect_training_rollout(
            policy,
            args,
            gamma=config.gamma,
            episode_idx=episode_idx,
            executor=executor,
        )
        if len(rollout) == 0:
            logger.warning("iter %d skipped empty rollout", episode_idx)
            continue

        if lr_annealing:
            lr_scale = lr_scale_at(args, episode_idx)
            for group, base in zip(trainer.optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_scale
        entropy_coef = entropy_coef_at(args, episode_idx)
        # Reuse recent rollouts (no-op when --replay-capacity 0); the current
        # rollout stays untouched so the logged diagnostics describe this episode.
        train_rollout = replay.combined(rollout)
        stats = trainer.update(train_rollout, entropy_coef=entropy_coef)
        replay.add(rollout)
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
        if eval_summary is not None:
            # Scenario-balanced, scale-free selection score (see checkpoint_score);
            # recorded in the log row and used to pick best.pt below.
            eval_summary["balanced_score"] = checkpoint_score(eval_summary)
        log_row = build_log_row(
            episode_idx=episode_idx,
            rollout=rollout,
            total_reward=total_reward,
            metrics=metrics,
            stats=stats,
            eval_summary=eval_summary,
            train_scenarios=training_scenario_counts(args, episode_idx),
            lr=trainer.optimizer.param_groups[0]["lr"],
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
            # Fall back to aggregate reward only when per-scenario baselines are
            # unavailable (consistent within a run, so still comparable).
            score = eval_summary["balanced_score"]
            if score is None:
                score = eval_summary["reward"]
            if score > best_eval_score:
                best_eval_score = score
                save_checkpoint(
                    torch=torch,
                    path=best_path,
                    episode_idx=episode_idx,
                    policy=policy,
                    trainer=trainer,
                    config=config,
                    args=args,
                    best_eval_reward=best_eval_score,
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
                best_eval_reward=best_eval_score,
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
) -> tuple[RolloutBuffer, dict[str, Any]]:
    """Collect ``rollout_episodes`` episodes and merge them in seed order.

    Merging in seed order (not completion order) keeps episode ids and joint
    indices identical to sequential collection, so credit assignment is
    unchanged regardless of which worker finished first. Returns the merged
    buffer and the per-field **mean** of the episode metrics across all rollout
    episodes, so the logged done/throughput/turnaround describe the whole batch
    instead of a single (last-seed) episode.
    """

    seeds = rollout_seeds(args, episode_idx)
    scenarios = rollout_scenarios(args, episode_idx)
    rollout = RolloutBuffer()
    episode_metrics: list[Any] = []

    if executor is None:
        for seed, scenario in zip(seeds, scenarios, strict=True):
            env = make_env(args, seed=seed, workload_scenario=scenario)
            episode_buffer = collect_episode(env, policy, seed=seed, gamma=gamma)
            episode_buffer.episode_scenarios[0] = scenario.value
            rollout.extend(episode_buffer)
            episode_metrics.append(env.metrics())
        return rollout, aggregate_episode_metrics(episode_metrics)

    cpu_state = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    tasks = [
        (cpu_state, seed, gamma, scenario)
        for seed, scenario in zip(seeds, scenarios, strict=True)
    ]
    for (_, _, _, scenario), (buffer, metrics) in zip(
        tasks, executor.map(_run_rollout_worker, tasks), strict=True
    ):
        # Worker buffers carry one episode (id 0); tag it before the offset merge.
        buffer.episode_scenarios[0] = scenario.value
        rollout.extend(buffer)
        episode_metrics.append(metrics)
    return rollout, aggregate_episode_metrics(episode_metrics)


def aggregate_episode_metrics(metrics_list: list[Any]) -> dict[str, Any]:
    """Mean of each ``EpisodeMetrics`` field across the rollout episodes.

    Task counts become floats (e.g. ``31.4``). ``None`` fields (percentiles for
    an episode that completed nothing) are dropped from the mean; a field that
    is ``None`` in every episode stays ``None``. ``per_core_utilization`` is
    averaged per core. Empty input returns an empty dict.
    """
    if not metrics_list:
        return {}
    return _mean_nested([metrics.as_dict() for metrics in metrics_list])


def _mean_nested(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if isinstance(values[0], dict):
            result[key] = _mean_nested(values)
        else:
            present = [value for value in values if value is not None]
            result[key] = float(np.mean(present)) if present else None
    return result


def entropy_coef_at(args: argparse.Namespace, episode_idx: int) -> float:
    """Linearly annealed entropy coefficient for ``episode_idx`` (1-based).

    Returns the constant ``--entropy-coef`` when ``--entropy-coef-final`` is
    unset. Otherwise interpolates from the initial to the final coefficient over
    ``--entropy-anneal-episodes`` (defaulting to the full ``--episodes`` budget)
    and holds the final value afterwards.
    """

    start = float(getattr(args, "entropy_coef", 0.0))
    final = getattr(args, "entropy_coef_final", None)
    if final is None:
        return start
    final = float(final)
    anneal_episodes = getattr(args, "entropy_anneal_episodes", None) or getattr(
        args, "episodes", 1
    )
    anneal_episodes = max(1, int(anneal_episodes))
    progress = min(1.0, max(0.0, (episode_idx - 1) / max(1, anneal_episodes - 1)))
    return start + (final - start) * progress


def lr_scale_at(args: argparse.Namespace, episode_idx: int) -> float:
    """Linear learning-rate multiplier (1.0 -> ``--lr-anneal-final-frac``).

    Returns 1.0 when annealing is disabled (final fraction == 1.0). Otherwise
    interpolates from 1.0 to the final fraction over ``--lr-anneal-episodes``
    (defaulting to the full ``--episodes`` budget) and holds it afterwards. Same
    schedule shape as :func:`entropy_coef_at`.
    """

    final_frac = float(getattr(args, "lr_anneal_final_frac", 1.0))
    if final_frac == 1.0:
        return 1.0
    anneal_episodes = getattr(args, "lr_anneal_episodes", None) or getattr(
        args, "episodes", 1
    )
    anneal_episodes = max(1, int(anneal_episodes))
    progress = min(1.0, max(0.0, (episode_idx - 1) / max(1, anneal_episodes - 1)))
    return 1.0 + (final_frac - 1.0) * progress


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
    per_scenario_rows = []
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
            per_scenario_rows.append(
                {
                    "scenario": scenario.value,
                    "reward": rollout.total_env_reward,
                    "completed": metrics.completed_tasks,
                    "throughput": metrics.throughput,
                    "preemptions": rollout.preemptions,
                    "turnaround": metrics.mean_turnaround_time,
                }
            )

    return {
        "reward": float(np.mean(rewards)),
        "completed": float(np.mean(completed)),
        "throughput": float(np.mean(throughputs)),
        "preemptions": float(np.mean(preemptions)),
        "scenario_counts": count_labels(scenario_labels),
        "by_scenario": summarize_eval_rows(per_scenario_rows),
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
        rows = []
        results = []
        for offset in range(episodes):
            scenario = scenarios[offset % len(scenarios)]
            result = run_episode(
                make_env(
                    args,
                    seed=base_seed + offset,
                    workload_scenario=scenario,
                ),
                baseline,
                seed=base_seed + offset,
            )
            results.append(result)
            rows.append(
                {
                    "scenario": scenario.value,
                    "reward": result.total_reward,
                    "completed": result.metrics.completed_tasks,
                    "throughput": result.metrics.throughput,
                    "turnaround": result.metrics.mean_turnaround_time,
                }
            )
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
            "by_scenario": summarize_eval_rows(rows),
        }
    return summaries


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    summaries: dict[str, dict[str, float]] = {}
    for scenario, scenario_rows in by_scenario.items():
        keys = [
            key
            for key, value in scenario_rows[0].items()
            if key != "scenario" and value is not None
        ]
        summaries[scenario] = {
            key: float(
                np.mean([
                    row[key]
                    for row in scenario_rows
                    if row.get(key) is not None
                ])
            )
            for key in keys
        }
        # Per-scenario reward spread + count, so consumers can form the standard
        # error (std/sqrt(n)) and judge whether the sample is large enough for an
        # RL-vs-baseline comparison to be meaningful (see summarize_scenario_
        # significance in plot_metrics).
        rewards = [row["reward"] for row in scenario_rows if row.get("reward") is not None]
        summaries[scenario]["n"] = len(rewards)
        summaries[scenario]["reward_std"] = (
            float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        )
    return summaries


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


CHECKPOINT_SCORE_CLAMP = 1.0


def checkpoint_score(eval_summary: dict[str, Any]) -> float | None:
    """Scenario-balanced, scale-free score for picking the best checkpoint.

    The aggregate eval reward is dominated by the highest-magnitude scenario
    (burst_stress rewards are ~6x balanced), so selecting ``best.pt`` on it
    biases the saved policy toward that one scenario. Instead, for each scenario
    we score the deterministic policy by its reward gap to the strongest
    *realistic* (non-oracle) baseline, as a fraction of that baseline's magnitude::

        rel = (rl - best_realistic_baseline) / |best_realistic_baseline|   (clamped +/-1)

    ``rel=0`` means "matched the best fair heuristic", ``>0`` beat it by that
    fraction, ``<0`` behind it. SJF-like is excluded because it is clairvoyant
    (an oracle ceiling, not a fair target -- see ``ORACLE_BASELINES``). Dividing
    by ``|best|`` (always large) keeps the score stable even when baselines bunch
    together. Scenarios are weighted equally, and because the denominator scales
    with the scenario a fixed reward delta counts more in a low-magnitude scenario
    (balanced) than a high one (burst) -- matching how much it actually means. The
    clamp stops one catastrophic scenario from swamping the mean. Returns ``None``
    when per-scenario baseline data is unavailable (caller falls back to aggregate).
    """
    by_scenario = eval_summary.get("by_scenario") or {}
    baselines = eval_summary.get("baselines") or {}
    if not by_scenario or not baselines:
        return None

    rels: list[float] = []
    for scenario, row in by_scenario.items():
        rl_reward = row.get("reward")
        if rl_reward is None:
            continue
        baseline_rewards = [
            value
            for name, entry in baselines.items()
            if name not in ORACLE_BASELINES
            and (value := entry.get("by_scenario", {}).get(scenario, {}).get("reward"))
            is not None
        ]
        if not baseline_rewards:
            continue
        best_baseline = max(baseline_rewards)
        scale = abs(best_baseline)
        if scale < 1e-9:
            continue
        rel = (rl_reward - best_baseline) / scale
        rels.append(max(-CHECKPOINT_SCORE_CLAMP, min(CHECKPOINT_SCORE_CLAMP, rel)))

    if not rels:
        return None
    return float(np.mean(rels))


def build_log_row(
    *,
    episode_idx: int,
    rollout,
    total_reward: float,
    metrics,
    stats,
    eval_summary: dict[str, float] | None,
    train_scenarios: dict[str, int] | None = None,
    lr: float | None = None,
) -> dict[str, Any]:
    elapsed_times = [transition.elapsed_time for transition in rollout.transitions]
    return {
        "episode": episode_idx,
        "lr": lr,
        "transitions": len(rollout),
        "joint_intervals": len(rollout.joint_transitions),
        "env_steps": rollout.env_steps,
        "conflicts": rollout.conflicts,
        "invalid_actions": rollout.invalid_actions,
        "preemptions": rollout.preemptions,
        "decisions": rollout.decisions,
        "noop_decisions": rollout.noop_decisions,
        "noop_fraction": rollout.noop_fraction,
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
        "metrics": metrics,
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
    get_logger().info("saved checkpoint=%s iter=%d", path, episode_idx)


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
