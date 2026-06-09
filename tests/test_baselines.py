from src.baselines import (
    EASLikePolicy,
    MLFQPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    SJFLikePolicy,
    run_episode,
)
from src.evaluate_baselines import EvalConfig, _make_env, _summarize_results
from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.env.task import LatencyClass, Task


def test_baseline_policies_finish_episode() -> None:
    policies = [
        RandomPolicy(seed=0),
        RoundRobinPolicy(),
        MLFQPolicy(),
        SJFLikePolicy(),
        EASLikePolicy(),
    ]

    for policy in policies:
        env = SchedulerEnv(
            core_config={CoreType.P: 1, CoreType.E: 1},
            workload_scenario=WorkloadScenario.BALANCED,
            arrival_rate=0.5,
            episode_time=40.0,
            max_tasks=8,
            seed=5,
        )

        result = run_episode(env, policy, seed=5)

        assert result.terminated
        assert not result.truncated
        assert result.metrics.completed_tasks == result.metrics.total_tasks
        assert result.steps > 0


def test_sjf_like_policy_selects_shortest_runtime_task() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=10.0,
        episode_time=10.0,
        max_tasks=4,
        seed=1,
    )
    observations, _ = env.reset()
    assert len(env.ready_queue) >= 1
    env.ready_queue.sort(key=lambda task: task.current_cpu_burst, reverse=True)

    actions = SJFLikePolicy().act(env, observations)
    chosen_task = env.ready_queue[actions["p_0"] - 1]

    assert chosen_task.current_cpu_burst == min(
        task.current_cpu_burst for task in env.ready_queue[: env.queue_size]
    )


def test_mlfq_policy_prefers_less_served_task() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=10.0,
        episode_time=10.0,
        max_tasks=4,
        seed=2,
    )
    observations, _ = env.reset()
    env.ready_queue = [
        _ready_task(pid=10, now=env.sim.now, cpu_progress=20.0),
        _ready_task(pid=11, now=env.sim.now, cpu_progress=0.0),
    ]

    actions = MLFQPolicy().act(env, observations)
    chosen_task = env.ready_queue[actions["p_0"] - 1]

    assert chosen_task.cpu_progress == 0.0


def test_mlfq_policy_aging_promotes_long_waiting_task() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=10.0,
        episode_time=10.0,
        max_tasks=4,
        seed=3,
    )
    observations, _ = env.reset()
    now = env.sim.now
    env.ready_queue = [
        _ready_task(pid=10, now=now - 100.0, cpu_progress=20.0),
        _ready_task(pid=11, now=now, cpu_progress=0.0),
    ]

    actions = MLFQPolicy().act(env, observations)
    chosen_task = env.ready_queue[actions["p_0"] - 1]

    assert chosen_task.pid == env.ready_queue[0].pid


def _ready_task(pid: int, now: float, cpu_progress: float) -> Task:
    task = Task(
        pid=pid,
        arrival_time=now,
        cpu_intensity=0.5,
        latency_class=LatencyClass.BEST_EFFORT,
        cpu_bursts=[10.0],
        cpu_progress=cpu_progress,
    )
    task.mark_ready(now)
    return task


def test_baseline_eval_summary_contains_mean_std_strings() -> None:
    config = EvalConfig(seeds=(0, 1), arrival_rate=0.4, episode_time=20.0, max_tasks=6)
    results = [
        run_episode(
            _make_env(WorkloadScenario.BALANCED, seed, config),
            RandomPolicy(seed=seed),
            seed=seed,
        )
        for seed in config.seeds
    ]

    row = _summarize_results("balanced", "random", results)

    assert row["scenario"] == "balanced"
    assert row["policy"] == "random"
    assert "+/-" in row["throughput"]
    assert "+/-" in row["mean_turnaround"]
