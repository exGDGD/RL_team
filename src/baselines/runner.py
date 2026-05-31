from __future__ import annotations

from dataclasses import dataclass

from src.env import EpisodeMetrics, SchedulerEnv

from .policies import BaselinePolicy


@dataclass(frozen=True)
class EpisodeResult:
    policy_name: str
    steps: int
    total_reward: float
    metrics: EpisodeMetrics
    reward_diagnostics: dict[str, float]
    terminated: bool
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy_name,
            "steps": self.steps,
            "total_reward": self.total_reward,
            **self.metrics.as_dict(),
            "reward_diagnostics": self.reward_diagnostics,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def run_episode(
    env: SchedulerEnv,
    policy: BaselinePolicy,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> EpisodeResult:
    policy.reset()
    observations, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    for step_idx in range(1, max_steps + 1):
        actions = policy.act(env, observations)
        observations, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += sum(rewards.values())
        terminated = all(terminations.values())
        truncated = all(truncations.values())
        if terminated or truncated:
            return EpisodeResult(
                policy_name=policy.name,
                steps=step_idx,
                total_reward=total_reward,
                metrics=env.metrics(),
                reward_diagnostics=env.reward_diagnostics(),
                terminated=terminated,
                truncated=truncated,
            )

    return EpisodeResult(
        policy_name=policy.name,
        steps=max_steps,
        total_reward=total_reward,
        metrics=env.metrics(),
        reward_diagnostics=env.reward_diagnostics(),
        terminated=False,
        truncated=True,
    )
