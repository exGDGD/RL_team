from __future__ import annotations

from typing import Protocol

from src.env import SchedulerEnv

from .buffer import AgentTransition, PendingDecision, RolloutBuffer
from .obs import AgentBatch, build_agent_batch


class RolloutPolicy(Protocol):
    """Policy interface used by the async rollout collector."""

    def act(
        self,
        batch: AgentBatch,
    ) -> (
        tuple[dict[str, int], dict[str, float]]
        | tuple[dict[str, int], dict[str, float], dict[str, object]]
    ):
        """Return actions, log-probs, and optionally effective action masks."""


def collect_episode(
    env: SchedulerEnv,
    policy: RolloutPolicy,
    *,
    seed: int | None = None,
    max_env_steps: int = 10_000,
) -> RolloutBuffer:
    """Collect one episode as agent-centric asynchronous transitions."""

    observations, info = env.reset(seed=seed)
    batch = build_agent_batch(observations, agent_order=env.agents)
    pending: dict[str, PendingDecision] = {}
    buffer = RolloutBuffer()

    for _ in range(max_env_steps):
        actions = {agent_id: 0 for agent_id in env.agents}
        chosen_actions, log_probs, effective_masks = _unpack_policy_output(policy.act(batch))
        proposed_agent_ids: set[str] = set()

        for agent_id in batch.decision_agent_ids():
            agent_index = batch.agent_ids.index(agent_id)
            task_choices = int(batch.action_mask[agent_index, 1:].sum())
            buffer.decisions += 1
            buffer.total_task_choices += task_choices
            buffer.max_task_choices = max(buffer.max_task_choices, task_choices)
            if task_choices <= 1:
                buffer.forced_decisions += 1
            action = int(chosen_actions.get(agent_id, 0))
            if action == 0:
                continue
            if agent_id in pending:
                raise RuntimeError(f"Agent {agent_id} has an unfinished pending decision.")
            actions[agent_id] = action
            proposed_agent_ids.add(agent_id)
            pending[agent_id] = PendingDecision(
                agent_id=agent_id,
                agent_index=agent_index,
                obs=batch,
                action=action,
                log_prob=float(log_probs.get(agent_id, 0.0)),
                action_mask=effective_masks.get(agent_id, batch.action_mask[agent_index]).copy(),
                start_time=float(info["time"]),
            )

        next_observations, rewards, terminations, truncations, next_info = env.step(actions)
        buffer.env_steps += 1
        buffer.conflicts += len(next_info.get("conflicts", {}))
        buffer.invalid_actions += len(next_info.get("invalid_actions", {}))
        _discard_rejected_decisions(
            pending=pending,
            proposed_agent_ids=proposed_agent_ids,
            assigned_agent_ids=set(next_info.get("assignments", {})),
        )
        next_batch = build_agent_batch(next_observations, agent_order=env.agents)
        terminated = all(terminations.values())
        truncated = all(truncations.values())

        _close_finished_decisions(
            pending=pending,
            buffer=buffer,
            rewards=rewards,
            finished_agent_ids={
                event["core_id"] for event in next_info.get("finished_runs", [])
            },
            next_batch=next_batch,
            next_time=float(next_info["time"]),
            terminated=terminated,
            truncated=truncated,
        )

        batch = next_batch
        info = next_info
        if terminated or truncated:
            return buffer

    _close_finished_decisions(
        pending=pending,
        buffer=buffer,
        rewards={agent_id: 0.0 for agent_id in env.agents},
        finished_agent_ids=set(),
        next_batch=batch,
        next_time=float(info["time"]),
        terminated=False,
        truncated=True,
    )
    return buffer


def _unpack_policy_output(policy_output):
    if len(policy_output) == 2:
        actions, log_probs = policy_output
        return actions, log_probs, {}
    actions, log_probs, effective_masks = policy_output
    return actions, log_probs, effective_masks


def _discard_rejected_decisions(
    *,
    pending: dict[str, PendingDecision],
    proposed_agent_ids: set[str],
    assigned_agent_ids: set[str],
) -> None:
    for agent_id in proposed_agent_ids - assigned_agent_ids:
        del pending[agent_id]


def _close_finished_decisions(
    *,
    pending: dict[str, PendingDecision],
    buffer: RolloutBuffer,
    rewards: dict[str, float],
    finished_agent_ids: set[str],
    next_batch: AgentBatch,
    next_time: float,
    terminated: bool,
    truncated: bool,
) -> None:
    for agent_id, decision in list(pending.items()):
        next_agent_index = next_batch.agent_ids.index(agent_id)
        agent_is_idle = next_batch.self_features[next_agent_index, 1] == 0.0
        finished_run = agent_id in finished_agent_ids
        if not (terminated or truncated or agent_is_idle or finished_run):
            continue

        buffer.append(
            AgentTransition(
                agent_id=agent_id,
                agent_index=decision.agent_index,
                obs=decision.obs,
                action=decision.action,
                log_prob=decision.log_prob,
                action_mask=decision.action_mask,
                reward=float(rewards.get(agent_id, 0.0)),
                next_obs=next_batch,
                next_agent_index=next_agent_index,
                elapsed_time=max(0.0, next_time - decision.start_time),
                terminated=terminated,
                truncated=truncated,
            )
        )
        del pending[agent_id]
