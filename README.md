# RL Team

Heterogeneous CPU scheduling을 위한 multi-agent reinforcement learning 실험 repo입니다.

초기 구현 목표는 SimPy 기반 이산 이벤트 시뮬레이터를 만들고, PettingZoo multi-agent API로 감싸서 학습/평가 코드와 연결하는 것입니다.

## Environment Setup

팀 공통 환경은 conda로 관리합니다. 환경 이름은 `rl-team`입니다.

```bash
conda env create -f environment.yml
conda activate rl-team
```

이미 환경을 만든 뒤 `environment.yml`이 바뀌었다면 다음 명령으로 갱신합니다.

```bash
conda env update -f environment.yml --prune
conda activate rl-team
```

설치가 잘 되었는지 확인합니다.

```bash
python -c "import simpy, pettingzoo, gymnasium, numpy; print('env ok')"
pytest -q
```

`environment.yml`은 Python과 pip 환경을 만들고, 실제 Python 패키지는 `requirements.txt`에서 설치합니다. 이렇게 두면 conda solver가 RL 패키지 의존성 해석에 오래 걸리는 문제를 줄일 수 있습니다.

### Optional venv Setup

conda를 쓰기 어려운 환경에서는 Python 3.12 기준으로 venv를 사용할 수 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`.venv/`는 각자 로컬에서만 생성하고 git에는 올리지 않습니다.

## Planned Stack

- `simpy`: task arrival, CPU burst, I/O wait 같은 비동기 이벤트 시뮬레이션
- `pettingzoo`: multi-agent environment API
- `gymnasium.spaces`: observation/action space 정의
- `numpy`: state/reward/metric 계산
- `pytest`: 환경 동작 단위 테스트

## Current Structure

```text
src/env/
  core.py           # heterogeneous core specs and runtime state
  metrics.py        # throughput, energy, latency, starvation, utilization metrics
  spaces.py         # Gymnasium observation/action space definitions
  task.py           # task phases, latency class, progress tracking
  workload.py       # stochastic task generator
  scheduler_env.py  # first-pass SimPy event-driven scheduler env
tests/
  test_baselines.py
  test_metrics.py
  test_scheduler_env.py
```

현재 `SchedulerEnv`는 PettingZoo `ParallelEnv`와 비슷한 dict 기반 입출력을 반환합니다. 정식 PettingZoo wrapper는 core simulator가 안정화된 뒤 얹을 예정입니다.

## SchedulerEnv Overview

`SchedulerEnv`는 SimPy 기반 이산 이벤트 시뮬레이터입니다. `reset()`에서 stochastic workload trace를 만들고, task arrival 또는 I/O completion으로 ready queue에 작업이 생긴 뒤 idle core가 있으면 decision point에서 멈춥니다.

`step(actions)`는 agent/core별 action을 받아 ready queue task를 core에 배정하고, CPU burst completion, I/O wait, 다음 task arrival 중 가장 이른 이벤트까지 simulated time을 진행합니다. 이 과정을 episode가 terminate/truncate될 때까지 반복합니다.

## Environment Semantics

- `episode_time`은 task arrival horizon입니다. 이 시점 이후 새 task는 생성하지 않지만, 이미 생성된 task는 완료될 때까지 drain합니다.
- `max_sim_time`은 무한 루프/비정상 trace를 막기 위한 truncation guard입니다.
- reward mode는 세 가지를 지원합니다. 기본값인 `event_shaped`는 CPU burst마다 CPU work progress reward와 energy/starvation cost를 주고, task 완료 시 task completion/work reward와 latency penalty를 줍니다. `event_cost`는 progress reward 없이 event cost와 completion reward만 사용합니다. `completion_only`는 비용을 누적했다가 task 완료 시 모든 reward를 한 번에 줍니다.
- completion reward는 task 개수 보상과 CPU work량 보상을 분리합니다. I/O wait은 completion reward 크기에 직접 포함하지 않습니다.
- latency penalty는 task 전체 완료 지연인 turnaround time 기준입니다.
- action은 step 시작 시점의 ready queue snapshot을 기준으로 해석합니다.
- 여러 idle core가 같은 task를 고르면 agent order가 빠른 core만 배정받고, 뒤의 중복 선택은 conflict/NO-OP 처리합니다.

## Observation and Action Space

`SchedulerEnv.action_space(agent_id)`는 `Discrete(queue_size + 1)`입니다. `0`은 NO-OP이고, `1..queue_size`는 ready queue slot 선택입니다.

`SchedulerEnv.observation_space(agent_id)`는 다음 key를 가진 `gymnasium.spaces.Dict`입니다.

- `self`: `(5,)` core type, busy flag, current task elapsed time, accumulated energy, time since last decision
- `ready_queue`: `(queue_size, 4)` waiting time, CPU progress, latency class, CPU intensity
- `ready_mask`: `(queue_size,)`
- `other_cores`: `(num_cores - 1, 3)` core type, busy flag, current task elapsed time
- `system`: `(6,)` total core count, utilization, 4-way core type counts
- `action_mask`: `(queue_size + 1,)`

## Metrics

`SchedulerEnv.metrics()`와 `info["metrics"]`는 다음 episode-level 지표를 제공합니다.

- throughput
- total energy
- mean/p95/p99 response time
- mean/p95/p99 turnaround time
- mean/p95 ready wait time
- starvation rate
- mean utilization
- per-core utilization

## Baselines

현재 baseline policy는 `src/baselines/`에 있습니다.

- `RandomPolicy`: valid ready-queue slot 중 무작위 선택
- `RoundRobinPolicy`: ready-queue slot을 순환 선택
- `SJFLikePolicy`: 현재 CPU burst runtime이 가장 짧은 task 선택
- `EASLikePolicy`: latency criticality, CPU intensity, core type affinity를 이용한 휴리스틱

간단한 multi-seed baseline 비교는 다음 명령으로 실행합니다. 출력값은 평균 `+/-` 표준편차 형식입니다.

```bash
python -m src.evaluate_baselines
```

## RL / Colab Smoke Test

로컬 가상환경은 simulator, baseline, numpy 기반 RL interface 테스트를 가볍게 돌리는 용도로 둡니다. PyTorch actor/critic forward pass와 이후 학습 실험은 Colab에서 실행합니다.

Colab에서는 `notebooks/acac_colab_smoke.ipynb`를 열고 위에서부터 실행하면 됩니다. 노트북은 repo를 clone한 뒤 `tests/test_rl_*`를 실행하고, `TypeSharedActor`, `AgentCentricCritic`, async rollout collector, PPO/ACAC update step이 서로 연결되는지 확인합니다.

## Collaboration Flow

1. 작업 전 repo를 최신 상태로 맞춥니다.
2. `conda activate rl-team`으로 가상환경을 활성화합니다.
3. 구현/테스트 후 필요한 파일만 commit합니다.
4. `.venv`, generated trace, training output은 commit하지 않습니다.

## notes
Main experiment v0(현재 버전):
  Global ready queue
  목적: heterogeneous core-task assignment 정책 검증

Extension / ablation v1:
  Per-core local queue + work stealing
  목적: 더 현실적인 OS scheduling 구조에서 정책이 유지되는지 검증
