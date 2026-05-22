import pytest

from src.env import Core, CoreType, LatencyClass, Task, compute_episode_metrics


def test_compute_episode_metrics_from_completed_tasks_and_cores() -> None:
    task_a = Task(
        pid=0,
        arrival_time=0.0,
        cpu_intensity=0.5,
        latency_class=LatencyClass.BEST_EFFORT,
        cpu_bursts=[1.0],
    )
    task_a.total_ready_wait_time = 2.0
    task_a.max_ready_wait_time = 2.0
    task_a.completed_at = 10.0

    task_b = Task(
        pid=1,
        arrival_time=5.0,
        cpu_intensity=0.5,
        latency_class=LatencyClass.SOFT_RT,
        cpu_bursts=[1.0],
    )
    task_b.total_ready_wait_time = 12.0
    task_b.max_ready_wait_time = 12.0
    task_b.completed_at = 25.0

    core = Core(core_id="p_0", core_type=CoreType.P)
    core.busy_time = 20.0
    core.accumulated_energy = 100.0

    metrics = compute_episode_metrics(
        tasks=[task_a, task_b],
        cores={"p_0": core},
        now=50.0,
        starvation_threshold=10.0,
    )

    assert metrics.total_tasks == 2
    assert metrics.completed_tasks == 2
    assert metrics.throughput == pytest.approx(0.04)
    assert metrics.total_energy == pytest.approx(100.0)
    assert metrics.mean_response_time == pytest.approx(15.0)
    assert metrics.mean_ready_wait_time == pytest.approx(7.0)
    assert metrics.starvation_rate == pytest.approx(0.5)
    assert metrics.mean_utilization == pytest.approx(0.4)
    assert metrics.per_core_utilization == {"p_0": pytest.approx(0.4)}
