# Codebase Overview

This document is a compact guide to the public release. It focuses on the files
needed to understand, run, and evaluate the heterogeneous CPU scheduling
experiment.

## Project Shape

The repository implements an event-driven multi-agent RL experiment for CPU
scheduling on heterogeneous cores.

- One core is one agent.
- Tasks have arrival times, CPU bursts, optional I/O waits, CPU intensity, and a
  latency class.
- The simulator is a SimPy discrete-event environment.
- The RL implementation uses type-shared actors and an agent-centric critic in
  an ACAC/PPO-style training loop.
- The default experiment uses a fixed P2E2 core configuration and can mix the
  four workload scenarios with `--train-scenarios all`.

## Main Entry Points

```text
src/env/scheduler_env.py   # simulator
src/baselines/policies.py  # random, round-robin, MLFQ-like, SJF-like, EAS-like
src/rl/rollout.py          # episode collection
src/rl/trainer.py          # ACAC/PPO update
src/train_acac.py          # training entrypoint
src/evaluate_baselines.py  # baseline-only evaluation
src/eval_checkpoint.py     # held-out checkpoint evaluation
src/plot_metrics.py        # training curve plotting
```

Optional warm-start tooling:

```text
src/rl/imitation.py
src/train_sjf_imitation.py
```

The SJF imitation path is an optional diagnostic/warm-start path. It is not a
baseline and should not be mixed with from-scratch RL results without labeling.

## Simulator Layer

`src/env/` contains the environment and scheduling model.

- `core.py`: core types and speed/power/context-switch specs.
- `task.py`: task state, CPU/I/O phases, progress, preemption bookkeeping.
- `workload.py`: stochastic workload generation for the four scenarios.
- `spaces.py`: Gymnasium observation/action-space contracts.
- `metrics.py`: throughput, energy, response/turnaround, wait, starvation, and
  utilization metrics.
- `scheduler_env.py`: reset/step logic, ready queue, event advancement,
  dispatch, preemption, rewards, and observations.

The action space is `Discrete(queue_size + 1)`. Action `0` means no change:
idle NO-OP for idle cores, or keep-current-task for busy cores. Actions
`1..queue_size` select ready-queue slots. With preemption enabled, an eligible
busy core can use a task action to preempt and switch.

The actor observation hides exact burst length and remaining CPU work. This is
intentional: SJF-like baselines may use simulator internals, but the learned
policy must act from observable features.

## Reward Modes

The environment supports:

| Mode | Meaning |
|---|---|
| `event_shaped` | Progress reward plus event costs and completion terms. |
| `event_cost` | Event costs and completion terms without progress shaping. |
| `completion_only` | Costs are accumulated and emitted at completion. |
| `latency_flow` | Priority-weighted flow-time penalty for unfinished tasks. |

The public experiment focuses on `latency_flow`. It charges a dense penalty for
every task that has arrived but not completed. Higher latency classes receive
larger weights, and tasks waiting for first execution receive an additional
response-time multiplier. This makes idle/no-dispatch behavior immediately
costly and aligns the reward with turnaround/response-time quality.

## Baselines

`src/baselines/policies.py` defines five heuristic/reference policies.

- `RandomPolicy`: sanity baseline.
- `RoundRobinPolicy`: simple ready-slot rotation.
- `MLFQPolicy`: OS-style service-history and aging heuristic.
- `EASLikePolicy`: hand-crafted latency/intensity/core-affinity heuristic.
- `SJFLikePolicy`: oracle-style shortest runtime reference.

Important interpretation details:

- `MLFQPolicy` is not observation-fair. It uses task identity and accumulated
  service history.
- `EASLikePolicy` is not a faithful Linux EAS implementation. It is a compact
  heterogeneous-core heuristic.
- `SJFLikePolicy` uses hidden current-burst/runtime information unavailable to
  the actor. Treat it as an oracle-style reference, not a fair competitor.

## RL Data Flow

Training follows this loop:

1. `train_acac.py` creates a `SchedulerEnv`.
2. `collect_episode` in `src/rl/rollout.py` runs the policy in the environment.
3. `RolloutBuffer` stores agent-centric transitions and joint macro-timeline
   transitions.
4. `ACACTrainer` computes time-scaled GAE over the joint timeline.
5. Actor transitions receive the advantage associated with their macro-timestep.
6. PPO-style clipped updates train the type-shared actors and critic.
7. Evaluation runs deterministic and sampled policies against the same held-out
   scenario mix and cached baselines.

Deterministic evaluation uses argmax actions. Sampled evaluation samples from
the stochastic policy learned by PPO.

## Checkpoint Selection

`train_acac.py` records raw reward and a `balanced_score`.

`balanced_score` is a scenario-balanced, scale-free checkpoint metric. For each
scenario, it compares the deterministic RL reward against the best non-oracle
baseline:

```text
(rl - best_non_oracle_baseline) / abs(best_non_oracle_baseline)
```

SJF-like is excluded because it is oracle-style. Scenario scores are clipped and
averaged equally so that high-magnitude scenarios do not dominate selection.

Use raw per-scenario reward, turnaround, throughput, and response time alongside
`balanced_score` when writing results.

## Minimal Commands

Baseline evaluation:

```bash
python -m src.evaluate_baselines
```

Training:

```bash
python -m src.train_acac \
  --reward-mode latency_flow \
  --train-scenarios all \
  --eval-scenarios all \
  --test-scenarios all \
  --episodes 300 \
  --eval-every 10 \
  --rollout-episodes 16 \
  --output-dir outputs/acac_p2e2
```

Checkpoint evaluation:

```bash
python -m src.eval_checkpoint outputs/acac_p2e2/best.pt --episodes 50
```

Plotting:

```bash
python -m src.plot_metrics outputs/acac_p2e2/metrics.jsonl --out curves.png
```

Optional SJF imitation warm start:

```bash
python -m src.train_sjf_imitation \
  --device cpu \
  --output outputs/sjf_imitation/actors.pt
```

## Result Framing

This release should be interpreted as a working experiment pipeline and
proof-of-concept, not a final scheduler. The strongest defensible claim is that
`latency_flow` makes the scheduling task learnable and that the trained policy
can become competitive with non-oracle heuristic references. The main remaining
limitations are incomplete convergence, the gap to SJF-like oracle behavior, and
weaker robustness on burst-heavy workloads.
