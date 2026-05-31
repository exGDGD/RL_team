import numpy as np
import pytest

from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import agents_by_core_type, build_agent_batch, mask_logits


def test_build_agent_batch_preserves_agent_order_and_shapes() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )
    observations, _ = env.reset()

    batch = build_agent_batch(observations, agent_order=env.agents)

    assert batch.agent_ids == tuple(env.agents)
    assert batch.num_agents == 2
    assert batch.queue_size == env.queue_size
    assert batch.self_features.shape == (2, 5)
    assert batch.ready_queue.shape == (2, env.queue_size, 6)
    assert batch.ready_mask.shape == (2, env.queue_size)
    assert batch.other_cores.shape == (2, 1, 3)
    assert batch.other_core_mask.shape == (2, 1)
    assert batch.system.shape == (2, 6)
    assert batch.action_mask.shape == (2, env.queue_size + 1)
    assert batch.delta_t.shape == (2,)
    assert batch.decision_mask.shape == (2,)


def test_agent_batch_marks_only_idle_agents_with_ready_tasks_as_decision_agents() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 2},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=1.0,
        episode_time=20.0,
        max_tasks=4,
        seed=11,
    )
    observations, _ = env.reset()
    observations, _, _, _, _ = env.step({"p_0": 1, "p_1": 0})

    batch = build_agent_batch(observations, agent_order=env.agents)

    assert not bool(batch.decision_mask[env.agents.index("p_0")])
    if any(obs["action_mask"][1:].any() for obs in observations.values()):
        assert batch.decision_mask.dtype == np.bool_


def test_agents_by_core_type_returns_shared_actor_groups() -> None:
    env = SchedulerEnv(
        core_config={CoreType.PRIME: 1, CoreType.P: 2, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=7,
    )
    observations, _ = env.reset()

    groups = agents_by_core_type(build_agent_batch(observations, agent_order=env.agents))

    assert groups[CoreType.PRIME].tolist() == [0]
    assert groups[CoreType.P].tolist() == [1, 2]
    assert groups[CoreType.E].tolist() == [3]
    assert groups[CoreType.LP_E].tolist() == []


def test_mask_logits_sets_invalid_actions_to_large_negative_value() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    action_mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

    masked = mask_logits(logits, action_mask, invalid_value=-123.0)

    assert masked.tolist() == [[1.0, -123.0, 3.0], [-123.0, 5.0, -123.0]]


def test_build_agent_batch_rejects_missing_agent_observation() -> None:
    with pytest.raises(KeyError):
        build_agent_batch({"p_0": {"self": np.zeros(5)}}, agent_order=("p_0", "p_1"))
