from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import AgentBatch, collect_episode


class FirstValidPolicy:
    def act(self, batch: AgentBatch) -> tuple[dict[str, int], dict[str, float]]:
        actions = {}
        log_probs = {}
        for row, agent_id in enumerate(batch.agent_ids):
            valid = [idx for idx, is_valid in enumerate(batch.action_mask[row]) if idx > 0 and is_valid]
            actions[agent_id] = valid[0] if bool(batch.decision_mask[row]) and valid else 0
            log_probs[agent_id] = 0.0
        return actions, log_probs


def test_collect_episode_returns_agent_centric_transitions() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1, CoreType.E: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )

    buffer = collect_episode(env, FirstValidPolicy(), seed=3)

    assert len(buffer) > 0
    assert all(transition.agent_id in env.agents for transition in buffer.transitions)
    assert all(transition.elapsed_time >= 0.0 for transition in buffer.transitions)
    assert env.completed_tasks
