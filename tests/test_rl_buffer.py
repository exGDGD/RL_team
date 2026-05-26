import numpy as np
import pytest

from src.rl import compute_time_scaled_gae


def test_time_scaled_gae_matches_one_step_td_when_lambda_zero() -> None:
    advantages, returns = compute_time_scaled_gae(
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        values=np.array([0.5, 0.25], dtype=np.float32),
        next_values=np.array([0.25, 0.0], dtype=np.float32),
        delta_t=np.array([2.0, 1.0], dtype=np.float32),
        dones=np.array([False, True]),
        gamma=0.9,
        gae_lambda=0.0,
    )

    assert advantages[0] == pytest.approx(1.0 + 0.9**2 * 0.25 - 0.5)
    assert advantages[1] == pytest.approx(2.0 - 0.25)
    assert returns[0] == pytest.approx(advantages[0] + 0.5)
    assert returns[1] == pytest.approx(advantages[1] + 0.25)


def test_time_scaled_gae_recurses_with_time_discount() -> None:
    advantages, _ = compute_time_scaled_gae(
        rewards=np.array([1.0, 1.0], dtype=np.float32),
        values=np.array([0.0, 0.0], dtype=np.float32),
        next_values=np.array([0.0, 0.0], dtype=np.float32),
        delta_t=np.array([2.0, 1.0], dtype=np.float32),
        dones=np.array([False, True]),
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert advantages[1] == pytest.approx(1.0)
    assert advantages[0] == pytest.approx(1.0 + 0.5**2 * 1.0)


def test_time_scaled_gae_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        compute_time_scaled_gae(
            rewards=np.array([1.0, 2.0]),
            values=np.array([0.0]),
            next_values=np.array([0.0, 0.0]),
            delta_t=np.array([1.0, 1.0]),
            dones=np.array([False, False]),
        )
