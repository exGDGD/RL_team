# RL Team

Minimal public release for a course project on heterogeneous CPU scheduling with
multi-agent reinforcement learning.

The project models a system with performance and efficiency cores, then trains
one scheduling agent per core to choose tasks from a shared ready queue. The
current release is a proof-of-concept experiment: it includes the simulator,
heuristic baselines, ACAC/PPO-style training code, and evaluation utilities. It
does not claim to be a production scheduler or a full replacement for OS
schedulers.

## What Is Included

```text
src/
  env/                 # event-driven scheduling simulator
  baselines/           # random, round-robin, MLFQ-like, SJF-like, EAS-like
  rl/                  # rollout, buffers, actor/critic networks, trainer
  evaluate_baselines.py
  train_acac.py
  eval_checkpoint.py
  plot_metrics.py
  train_sjf_imitation.py
docs/
  codebase-overview.md
tests/
```

The default environment uses a P2E2 core configuration: two performance cores
and two efficiency cores. Four workload scenarios are available: `balanced`,
`ui_heavy`, `bg_heavy`, and `burst_stress`.

## Setup

The shared environment is managed with conda:

```bash
conda env create -f environment.yml
conda activate rl-team
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate rl-team
```

The base requirements cover the simulator, baselines, plotting, and most tests.
PyTorch is required for actor/critic training and is intentionally not pinned in
`requirements.txt` because CPU/CUDA installation differs by platform. Install a
matching PyTorch build before running `src.train_acac` or
`src.train_sjf_imitation`.

## Quick Checks

```bash
python -c "import simpy, gymnasium, numpy; print('env ok')"
python -m pytest -q
```

If PyTorch is not installed, the torch-dependent tests are skipped. If
`matplotlib` is unavailable in a local environment, the core test suite can be
checked with:

```bash
python -m pytest -q --ignore=tests/test_plot_metrics.py
```

## Simulator

`SchedulerEnv` is a SimPy-based discrete-event simulator. Each episode generates
a finite stochastic workload trace. Tasks have arrival times, CPU bursts,
optional I/O waits, CPU intensity, and a latency class. The simulator advances to
the next scheduling decision point, accepts per-core actions, and then advances
to the next arrival, I/O completion, CPU burst completion, or preemption-relevant
wakeup.

The action space is `Discrete(queue_size + 1)`:

- `0`: no change. For an idle core this is a NO-OP; for a busy core this keeps
  the current task.
- `1..queue_size`: select a ready-queue slot. If preemption is enabled and a
  busy core is eligible, selecting a slot preempts the current task and switches
  to the selected task.

The actor observation intentionally does not expose exact current burst length or
remaining CPU work. SJF-like baselines can use simulator internals, but the RL
actor only sees observable task features such as waiting time, progress, latency
class, and CPU intensity.

## Reward Modes

The simulator supports four reward modes:

- `event_shaped`: progress reward plus energy/starvation costs and completion
  terms.
- `event_cost`: event costs and completion terms without progress shaping.
- `completion_only`: costs are accumulated and emitted when a task completes.
- `latency_flow`: priority-weighted flow-time penalty for all arrived but
  unfinished tasks.

The main experiment uses `latency_flow`. In this mode, rewards are usually
negative and values closer to zero are better. A task accrues penalty while it is
in the system; higher latency classes receive larger weights, and tasks that
have not yet started receive an extra response-time multiplier. This prevents an
idle/no-dispatch policy from looking artificially good.

## Baselines

Run the built-in baseline comparison:

```bash
python -m src.evaluate_baselines
```

Baseline interpretation:

- `RandomPolicy` and `RoundRobinPolicy` are weak sanity baselines.
- `MLFQPolicy` is an OS-style heuristic reference. It uses task identity and
  accumulated service history, so it is not observation-fair with the RL actor.
- `EASLikePolicy` is a hand-crafted heterogeneous-core affinity heuristic. It is
  not a faithful Linux EAS implementation.
- `SJFLikePolicy` is an oracle-style reference. It uses hidden current-burst and
  core-runtime information that the actor does not observe.

For this reason, SJF-like should be read as a strong reference/ceiling, not as a
fair learned-policy competitor.

## Training

Minimal ACAC/PPO training command:

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

Useful stability options:

```bash
python -m src.train_acac \
  --reward-mode latency_flow \
  --train-scenarios all \
  --eval-scenarios all \
  --test-scenarios all \
  --episodes 300 \
  --rollout-episodes 32 \
  --advantage-norm per_scenario \
  --lr-anneal-final-frac 0.1 \
  --entropy-coef 0.01 \
  --entropy-coef-final 0.0
```

Training writes `metrics.jsonl`, `train.log`, `latest.pt`, and `best.pt` under
the output directory. `best.pt` is selected using `balanced_score`, a
scenario-balanced score against the best non-oracle baseline.

## Evaluation

Evaluate a saved checkpoint on held-out seeds:

```bash
python -m src.eval_checkpoint outputs/acac_p2e2/best.pt --episodes 50
```

Plot a training log:

```bash
python -m src.plot_metrics outputs/acac_p2e2/metrics.jsonl --out curves.png
```

Two policy execution modes are logged:

- deterministic evaluation: chooses the argmax action from the learned policy.
- sampled evaluation: samples from the stochastic PPO policy.

PPO trains a stochastic policy, so sampled evaluation reflects the learned
distribution directly. Deterministic evaluation is a greedy deployment-style
view. A small gap between the two suggests that the action distribution has
concentrated around the greedy behavior.

## `balanced_score`

`balanced_score` is used for checkpoint selection, not as a raw reward metric.
For each scenario, it computes the deterministic RL policy's relative reward gap
against the best realistic non-oracle baseline:

```text
(RL reward - best realistic baseline reward) / abs(best realistic baseline reward)
```

The score excludes SJF-like because SJF-like uses hidden runtime information.
Scenario scores are clipped to `[-1, 1]` and averaged equally so that high-scale
scenarios such as `burst_stress` do not dominate checkpoint selection.

Interpretation:

- `0`: matched the best non-oracle baseline on average.
- `> 0`: outperformed the best non-oracle baseline on average.
- `< 0`: underperformed the best non-oracle baseline on average.

Raw reward, turnaround, throughput, and per-scenario results should still be
reported alongside `balanced_score`.

## Optional SJF Imitation Warm Start

`src.train_sjf_imitation` is included as an optional diagnostic and warm-start
tool. It trains actor weights to imitate SJF-like labels generated from simulator
internals.

This path is:

- optional, not required for default training;
- not a baseline;
- not a purely from-scratch RL result;
- based on oracle-style SJF-like labels, so runs using it should be reported
  separately.

Example:

```bash
python -m src.train_sjf_imitation \
  --device cpu \
  --output outputs/sjf_imitation/actors.pt

python -m src.train_acac \
  --reward-mode latency_flow \
  --pretrained-actors outputs/sjf_imitation/actors.pt \
  --output-dir outputs/acac_sjf_warm_start
```

## Current Result Interpretation

The current experiments are best described as a competitive proof of concept.
`LATENCY_FLOW` provides a learnable signal, and the learned policy can become
competitive with non-oracle heuristics such as MLFQ-like and EAS-like. However,
the policy should not be presented as a superior scheduler in general. It may
plateau before reaching the SJF-like oracle reference, and `burst_stress`
workloads remain a key challenge for user-perceived responsiveness.

See `docs/codebase-overview.md` for a compact description of the implementation
and data flow.
