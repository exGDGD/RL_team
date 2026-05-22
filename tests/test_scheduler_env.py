from src.env import (
    CoreType,
    LatencyClass,
    RewardMode,
    RewardWeights,
    SchedulerEnv,
    Task,
    WorkloadScenario,
)
import pytest


def test_reset_returns_observations_for_all_cores() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.4,
        episode_time=50.0,
        max_tasks=8,
        seed=7,
    )

    observations, info = env.reset()

    assert set(observations) == {"p_0", "e_0"}
    assert info["time"] >= 0.0
    for obs in observations.values():
        assert obs["self"].shape == (5,)
        assert obs["ready_queue"].shape == (8, 4)
        assert obs["ready_mask"].shape == (8,)
        assert obs["action_mask"].shape == (9,)


def test_observations_and_actions_match_declared_spaces() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.4,
        episode_time=50.0,
        max_tasks=8,
        seed=7,
    )

    observations, _ = env.reset()

    for agent_id, observation in observations.items():
        assert env.observation_space(agent_id).contains(observation)
        assert env.action_space(agent_id).contains(0)
        assert env.action_space(agent_id).contains(env.queue_size)
        assert not env.action_space(agent_id).contains(env.queue_size + 1)


def test_unknown_agent_space_lookup_raises_key_error() -> None:
    env = SchedulerEnv(core_config={CoreType.P: 1})
    env.reset(seed=0)

    with pytest.raises(KeyError):
        env.observation_space("missing_core")


def test_environment_can_finish_with_first_valid_action_policy() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.UI_HEAVY,
        arrival_rate=0.5,
        episode_time=40.0,
        max_tasks=6,
        seed=3,
    )

    observations, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        actions = {}
        for agent_id, obs in observations.items():
            valid_task_actions = [
                idx for idx, is_valid in enumerate(obs["action_mask"]) if idx > 0 and is_valid
            ]
            actions[agent_id] = valid_task_actions[0] if valid_task_actions else 0

        observations, _, terminations, truncations, _ = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        steps += 1

    assert done
    assert env.completed_tasks
    assert steps < 100
    assert env.metrics().completed_tasks == len(env.completed_tasks)


def test_info_includes_episode_metrics() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.4,
        episode_time=50.0,
        max_tasks=8,
        seed=7,
    )

    _, info = env.reset()

    assert "metrics" in info
    assert info["metrics"]["total_tasks"] == info["total_tasks"]
    assert "throughput" in info["metrics"]
    assert "total_energy" in info["metrics"]


def test_duplicate_task_selection_is_resolved_as_conflict() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 2},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=11,
    )

    observations, _ = env.reset()
    assert observations["p_0"]["action_mask"][1] == 1
    assert observations["p_1"]["action_mask"][1] == 1

    _, _, _, _, info = env.step({"p_0": 1, "p_1": 1})

    assert info["assignments"] == {"p_0": 0}
    assert info["conflicts"] == {"p_1": 0}


def test_episode_time_is_arrival_horizon_not_truncation_time() -> None:
    env = SchedulerEnv(
        core_config={CoreType.LP_E: 1},
        workload_scenario=WorkloadScenario.BURST_STRESS,
        arrival_rate=1.0,
        episode_time=1.0,
        max_sim_time=500.0,
        max_tasks=1,
        seed=2,
    )

    observations, _ = env.reset()
    done = False
    truncation_seen = False
    while not done:
        actions = {
            agent_id: 1 if obs["action_mask"][1] else 0
            for agent_id, obs in observations.items()
        }
        observations, _, terminations, truncations, _ = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        truncation_seen = truncation_seen or all(truncations.values())

    assert all(terminations.values())
    assert not truncation_seen
    assert env.sim.now > env.workload_config.episode_time


def test_max_sim_time_truncates_unfinished_trace() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=100.0,
        episode_time=10.0,
        max_sim_time=0.0,
        max_tasks=1,
        seed=1,
    )

    observations, _ = env.reset()
    actions = {agent_id: 0 for agent_id in observations}
    _, _, terminations, truncations, _ = env.step(actions)

    assert not all(terminations.values())
    assert all(truncations.values())


def test_event_cost_reward_mode_emits_cost_before_task_completion() -> None:
    env = SchedulerEnv(
        reward_mode=RewardMode.EVENT_COST,
        reward_weights=RewardWeights(completion=10.0, energy=1.0, starvation=1.0, latency=1.0),
    )
    task = Task(
        pid=0,
        arrival_time=0.0,
        cpu_intensity=0.5,
        latency_class=LatencyClass.SOFT_RT,
        cpu_bursts=[1.0, 1.0],
        io_waits=[1.0],
    )

    reward = env._reward_for_finished_burst(
        task=task,
        energy_cost=2.0,
        starvation_cost=3.0,
    )

    assert reward == -5.0


def test_completion_only_reward_mode_waits_until_task_completion() -> None:
    env = SchedulerEnv(
        reward_mode=RewardMode.COMPLETION_ONLY,
        reward_weights=RewardWeights(completion=10.0, energy=1.0, starvation=1.0, latency=1.0),
    )
    task = Task(
        pid=0,
        arrival_time=0.0,
        cpu_intensity=0.5,
        latency_class=LatencyClass.SOFT_RT,
        cpu_bursts=[1.0, 1.0],
        io_waits=[1.0],
    )
    task.accumulate_costs(energy_cost=2.0, starvation_cost=3.0)

    mid_reward = env._reward_for_finished_burst(
        task=task,
        energy_cost=2.0,
        starvation_cost=3.0,
    )
    task.completed_at = 10.0
    completion_reward = env._reward_for_finished_burst(
        task=task,
        energy_cost=0.0,
        starvation_cost=0.0,
    )

    assert mid_reward == 0.0
    assert completion_reward == -5.0
