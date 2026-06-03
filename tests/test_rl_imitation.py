from src.env import CoreType, SchedulerEnv, WorkloadScenario
from src.rl import collect_sjf_examples


def test_collect_sjf_examples_records_learnable_masked_actions() -> None:
    env = SchedulerEnv(
        core_config={CoreType.P: 1},
        workload_scenario=WorkloadScenario.BALANCED,
        arrival_rate=2.0,
        episode_time=30.0,
        max_tasks=16,
        seed=3,
    )

    examples = collect_sjf_examples(env, seed=3)

    assert examples
    assert all(not example.action_mask[0] for example in examples)
    assert all(example.action_mask[example.action] for example in examples)
    assert all(example.action_mask.sum() > 1 for example in examples)
