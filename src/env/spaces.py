from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .core import CoreType


SELF_FEATURE_DIM = 5
READY_TASK_FEATURE_DIM = 4
OTHER_CORE_FEATURE_DIM = 3
SYSTEM_FEATURE_DIM = 2 + len(CoreType)


def build_action_space(queue_size: int) -> spaces.Discrete:
    """Action 0 is NO-OP; actions 1..queue_size select ready queue slots."""
    return spaces.Discrete(queue_size + 1)


def build_observation_space(
    *,
    queue_size: int,
    num_cores: int,
) -> spaces.Dict:
    other_core_count = max(0, num_cores - 1)
    max_core_type = len(CoreType) - 1
    return spaces.Dict(
        {
            "self": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array(
                    [max_core_type, 1.0, np.inf, np.inf, np.inf],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            "ready_queue": spaces.Box(
                low=np.zeros((queue_size, READY_TASK_FEATURE_DIM), dtype=np.float32),
                high=np.tile(
                    np.array([np.inf, np.inf, 2.0, 1.0], dtype=np.float32),
                    (queue_size, 1),
                ),
                dtype=np.float32,
            ),
            "ready_mask": spaces.MultiBinary(queue_size),
            "other_cores": spaces.Box(
                low=np.zeros((other_core_count, OTHER_CORE_FEATURE_DIM), dtype=np.float32),
                high=np.tile(
                    np.array([max_core_type, 1.0, np.inf], dtype=np.float32),
                    (other_core_count, 1),
                ),
                dtype=np.float32,
            ),
            "system": spaces.Box(
                low=np.zeros((SYSTEM_FEATURE_DIM,), dtype=np.float32),
                high=np.array(
                    [np.inf, 1.0, np.inf, np.inf, np.inf, np.inf],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(queue_size + 1),
        }
    )
