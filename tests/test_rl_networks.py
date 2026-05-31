import pytest

torch = pytest.importorskip("torch")

from src.rl.networks import AgentCentricCritic, TypeSharedActor


def test_type_shared_actor_outputs_masked_action_logits() -> None:
    actor = TypeSharedActor(hidden_dim=16)
    batch_size = 3
    queue_size = 5

    logits = actor(
        self_features=torch.zeros(batch_size, 5),
        ready_queue=torch.zeros(batch_size, queue_size, 6),
        ready_mask=torch.ones(batch_size, queue_size),
        other_cores=torch.zeros(batch_size, 2, 3),
        other_core_mask=torch.ones(batch_size, 2),
        system=torch.zeros(batch_size, 6),
        action_mask=torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )

    assert logits.shape == (batch_size, queue_size + 1)
    assert logits[0, 2].item() < -1.0e8
    assert logits[1, 1].item() < -1.0e8


def test_agent_centric_critic_outputs_one_value_per_agent() -> None:
    critic = AgentCentricCritic(hidden_dim=16, num_heads=4)
    batch_size = 3

    values = critic(
        self_features=torch.zeros(batch_size, 5),
        ready_queue=torch.zeros(batch_size, 5, 6),
        ready_mask=torch.ones(batch_size, 5),
        other_cores=torch.zeros(batch_size, 2, 3),
        other_core_mask=torch.ones(batch_size, 2),
        system=torch.zeros(batch_size, 6),
    )

    assert values.shape == (batch_size,)
