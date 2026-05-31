from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import AgentBatch, RolloutBuffer, collect_episode
from src.rl.rollout import _discard_rejected_decisions
from src.train_acac import summarize_rollout_actions


class FirstValidPolicy:
    def act(self, batch: AgentBatch) -> tuple[dict[str, int], dict[str, float]]:
        actions = {}
        log_probs = {}
        for row, agent_id in enumerate(batch.agent_ids):
            valid = [idx for idx, is_valid in enumerate(batch.action_mask[row]) if idx > 0 and is_valid]
            actions[agent_id] = valid[0] if bool(batch.decision_mask[row]) and valid else 0
            log_probs[agent_id] = 0.0
        return actions, log_probs


class NoOpPolicy:
    def act(self, batch: AgentBatch) -> tuple[dict[str, int], dict[str, float]]:
        return (
            {agent_id: 0 for agent_id in batch.agent_ids},
            {agent_id: 0.0 for agent_id in batch.agent_ids},
        )


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
    assert buffer.decisions > 0
    assert buffer.mean_task_choices >= 1.0
    assert 0.0 <= buffer.forced_decision_fraction <= 1.0
    assert env.completed_tasks


def test_collect_episode_does_not_store_noop_as_pending_macro_action() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )

    buffer = collect_episode(env, NoOpPolicy(), seed=3, max_env_steps=3)

    assert len(buffer) == 0


def test_rejected_conflict_decision_is_removed_from_pending() -> None:
    pending = {"p_0": object(), "p_1": object()}

    _discard_rejected_decisions(
        pending=pending,
        proposed_agent_ids={"p_0", "p_1"},
        assigned_agent_ids={"p_0"},
    )

    assert set(pending) == {"p_0"}


def test_combined_rollouts_receive_distinct_episode_ids() -> None:
    combined = RolloutBuffer()
    for seed in (3, 4):
        env = SchedulerEnv(
            core_config={CoreType.P: 1},
            workload_scenario=WorkloadScenario.BALANCED,
            arrival_rate=0.5,
            episode_time=30.0,
            max_tasks=4,
            seed=seed,
        )
        combined.extend(collect_episode(env, FirstValidPolicy(), seed=seed))

    assert combined.episodes == 2
    assert {transition.episode_id for transition in combined.transitions} == {0, 1}


def test_rollout_action_summary_describes_selected_task_features() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=0.5,
        episode_time=30.0,
        max_tasks=4,
        seed=3,
    )

    summary = summarize_rollout_actions(collect_episode(env, FirstValidPolicy(), seed=3))

    assert summary["mean_queue_slot"] == 1.0
    assert summary["first_slot_fraction"] == 1.0
    assert summary["mean_selected_wait"] >= 0.0
    assert 0.0 <= summary["mean_selected_latency"] <= 2.0
    assert 0.0 <= summary["mean_selected_cpu_intensity"] <= 1.0
