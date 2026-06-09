# RL Team

성능–효율 비대칭(heterogeneous) CPU 코어가 혼합된 환경에서 "어떤 태스크를 어느 코어에 보낼지"를 결정하는 스케줄러를 비동기 multi-agent reinforcement learning으로 학습하는 실험 repo입니다.

- **알고리즘:** ACAC (Agent-Centric Actor-Critic for Asynchronous MARL, ICML 2025). 1 core = 1 agent, 코어 타입별 정책 파라미터 공유.
- **장기 목표:** 한 번 학습한 정책이 다양한 코어 구성(P2E2, P1E3, P3E1…)에 zero-shot으로 작동하는 scalability. 이를 위해 domain randomization과 12 코어구성 × 5 워크로드 grid 평가를 계획합니다.
- **현재 상태:** SimPy 기반 이산 이벤트 시뮬레이터 + baseline 4종 + 단일 구성(P2E2) ACAC sanity 학습까지 구현된 단계입니다. domain randomization·평가 grid·replay trace는 미구현입니다.

설계 동기·환경 설계·검증 방법의 전체 맥락은 [`docs/teamplo-design-doc.md`](docs/teamplo-design-doc.md)에 정리되어 있습니다. 초기 구현 목표는 SimPy 시뮬레이터를 만들고 PettingZoo multi-agent API로 감싸 학습/평가 코드와 연결하는 것입니다.

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
- `torch`: actor/critic 학습 (requirements에는 없으며 Colab에서만 설치·실행)

## Heterogeneous Core Architecture

`src/env/core.py`의 `CORE_SPECS`는 4종 코어를 정의합니다. 실행시간은 `Δt = B_base / speed × α_mismatch`로 계산하고, mismatch penalty(`SchedulerEnv._mismatch_penalty`)는 HARD-RT 태스크의 E/LP-E 배정(×1.4), 고-intensity 태스크의 LP-E 배정(×1.5), 저-intensity 태스크의 Prime/P 배정(×1.15)에 적용됩니다.

| Core Type | 처리속도 배율 | 전력 계수 | Context-Switch 비용 | 주 용도 |
|---|---|---|---|---|
| Prime-Core | 4.0× | 8.0 | 1.5 | 단일 스레드 최고 성능 |
| P-Core | 3.0× | 5.0 | 1.0 | 헤비 병렬 연산 |
| E-Core | 1.5× | 1.5 | 0.3 | I/O, 인터럽트 |
| LP-E Core | 0.8× | 0.4 | 0.2 | 백그라운드 장기 태스크 |

기본 코어 구성은 **P2E2**(`DEFAULT_CORE_CONFIG = {P: 2, E: 2}`)입니다. Prime/LP-E는 스펙만 정의되어 있고 기본 학습/테스트에는 포함하지 않습니다.

## Current Structure

```text
src/
  env/
    core.py           # heterogeneous core specs and runtime state
    task.py           # multi-phase task (cpu_bursts/io_waits), latency class, progress
    workload.py       # stochastic task generator (scenario-conditioned)
    metrics.py        # throughput, energy, latency, starvation, utilization metrics
    spaces.py         # Gymnasium observation/action space definitions
    scheduler_env.py  # SimPy event-driven scheduler env (global ready queue)
  baselines/
    policies.py       # Random / RoundRobin / SJF-like / EAS-like
    runner.py         # baseline episode runner
  rl/
    obs.py            # observation tensor encoding
    networks.py       # TypeSharedActor + AgentCentricCritic (attention)
    buffer.py         # RolloutBuffer + joint macro-timeline transitions, time-scaled GAE
    rollout.py        # async episode collection
    trainer.py        # ACAC/PPO update step
    imitation.py      # SJF example collection for warm-start
  evaluate_baselines.py  # multi-seed baseline comparison
  train_acac.py          # single-config (P2E2) ACAC sanity training entrypoint
  train_sjf_imitation.py # SJF actor warm-start
  train_logging.py       # human-readable console/file training logs
tests/                    # test_baselines, test_metrics, test_scheduler_env, test_rl_*, test_train_acac
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

`SchedulerEnv.action_space(agent_id)`는 `Discrete(queue_size + 1)`입니다. `0`은 "변화 없음"(idle이면 NO-OP/대기, busy이면 현재 task 유지)이고, `1..queue_size`는 ready queue slot 선택입니다. 선점이 켜져 있으면 busy core가 slot을 고르면 현재 task를 preempt하고 그 slot으로 전환합니다.

`SchedulerEnv.observation_space(agent_id)`는 다음 key를 가진 `gymnasium.spaces.Dict`입니다.

- `self`: `(8,)` core type, busy flag, current task elapsed time, accumulated energy, time since last decision, running task latency class, running task CPU intensity, running task CPU progress
- `ready_queue`: `(queue_size, 4)` waiting time, CPU progress, latency class, CPU intensity (정확한 burst 길이는 SJF 정답지화를 막기 위해 노출하지 않음)
- `ready_mask`: `(queue_size,)`
- `other_cores`: `(num_cores - 1, 3)` core type, busy flag, current task elapsed time
- `system`: `(6,)` total core count, utilization, 4-way core type counts
- `action_mask`: `(queue_size + 1,)`

> **선점(preemption):** `enable_preemption=True`(학습 기본값)이면 wakeup(arrival/IO 완료) 시 우선순위(P1)·기아(P3) 게이트를 통과한 busy core도 decision point를 받아 현재 task를 멈추고 다른 task로 전환할 수 있습니다(부분 burst 재큐잉 + 코어별 context-switch 비용). 비선점 ablation은 `--disable-preemption`. 자세한 설계는 [`docs/preemption-design.md`](docs/preemption-design.md).

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

Single-config sanity training은 Colab에서 다음처럼 실행합니다.

```bash
python -m src.train_acac --episodes 100 --eval-every 10
```

현재 학습 entrypoint는 `P2E2 + balanced workload` 고정 구성으로 시작합니다. 기본 arrival rate는 `1.0`, 최대 task 수는 `64`입니다. 너무 한산한 workload에서는 대부분의 decision에 선택 가능한 task가 하나뿐이라 정책을 학습할 수 없습니다. 출력의 `choices`와 `forced`를 함께 확인합니다. `SchedulerEnv`의 NO-OP는 아직 idle duration을 진행시키지 않으므로, 초기 ACAC sanity training에서는 NO-OP sampling을 비활성화합니다.

학습 update 한 번에는 기본적으로 16개 episode rollout을 합칩니다. ACAC critic은 누군가 새 scheduling decision을 만드는 시점을 shared joint macro-timestep으로 사용합니다. 각 joint interval 내부의 system-wide team reward는 simulated elapsed time에 따라 할인하고 agent 수로 평균냅니다. Joint timeline에서 GAE를 한 번 계산한 뒤, 각 actor action은 자신이 시작된 joint macro-timestep의 advantage를 사용합니다. 선택 가능한 task가 하나뿐인 forced transition은 critic timeline에는 남기되 actor update에서는 제외합니다. 콘솔과 평가의 reward는 학습용 평균 reward가 아니라 환경이 실제로 방출한 episode 총점입니다. critic target은 raw 평가 reward와 분리하여 `reward_scale=0.01`을 적용하고, actor와 critic gradient clipping도 별도로 수행합니다. 입력 observation의 대기시간, 진행시간, 누적 에너지는 MLP에 넣기 전에 `log1p`로 안정화합니다.

현재 구현은 joint macro-timeline GAE까지 반영한 단계입니다. 공식 ACAC의 GRU history encoder, time embedding, PopArt value normalization, target critic은 이후 단계에서 추가합니다.

PPO를 조정하기 전에 actor 표현력과 action-mask 경로를 분리해서 확인하려면 SJF imitation sanity test를 실행합니다. 이 학습은 ready queue의 current CPU burst를 직접 관측하고 SJF label을 지도학습합니다.

```bash
python -m src.train_sjf_imitation \
  --device cuda \
  --output outputs/sjf_imitation/actors.pt
```

저장된 actor를 PPO 초기값으로 사용할 수도 있습니다.

```bash
python -m src.train_acac \
  --device cuda \
  --pretrained-actors outputs/sjf_imitation/actors.pt \
  --output-dir outputs/acac_sjf_warm_start
```

```bash
python -m src.train_acac \
  --rollout-episodes 16 \
  --reward-scale 0.01 \
  --actor-learning-rate 0.0003 \
  --critic-learning-rate 0.0003 \
  --clip-ratio 0.05 \
  --entropy-coef 0.0 \
  --update-epochs 2
```

Energy와 latency cost는 물리적으로 해석 가능한 raw 값을 유지하고 lambda로 trade-off를 조정합니다. Starvation cost는 queue 길이와 긴 대기시간 때문에 폭발하지 않도록 `mean(log1p(wait)) + beta * max(log1p(wait))`에 burst 실행 시간을 곱합니다. 초기 sanity training은 모든 task가 완료되는 trace를 사용하므로 episode 합에서 거의 상수인 progress/completion shaping은 끕니다. 정책 차이가 충분히 보이도록 `progress=0.0`, `completion=0.0`, `completion_work=0.0`, `energy=0.1`, `starvation=0.05`, `latency=0.5`, `beta=0.5`를 사용합니다. Cost-only reward는 0에 가까울수록 좋습니다.

```bash
python -m src.train_acac \
  --progress-work 0.0 \
  --completion 0.0 \
  --completion-work 0.0 \
  --lambda-energy 0.1 \
  --lambda-starvation 0.05 \
  --lambda-latency 0.5 \
  --starvation-max-wait-weight 0.5
```

학습 중 `outputs/acac_p2e2/`에 `metrics.jsonl`, `train.log`, `latest.pt`, `best.pt`가 생성됩니다. Colab 런타임 종료 후에도 보존하려면 `--output-dir`에 Google Drive 경로를 넘깁니다. 중단된 학습은 다음처럼 이어서 실행합니다.

로그는 두 갈래로 나뉩니다. `metrics.jsonl`은 episode별 전체 지표를 담는 기계 판독용 구조화 기록이고, 콘솔과 `train.log`(`src/train_logging.py`)는 사람이 한눈에 훑기 위한 요약입니다. 매 episode `ep ... | reward ... | loss ...` 한 줄이 찍히고, 평가 episode에서는 그 아래로 `eval`(deterministic / sampled reward)과 `base`(random/sjf/eas baseline) 블록이 들여쓰여 추가됩니다. eval 줄의 `reward`는 argmax action을 쓰는 deterministic 평가, 괄호 안 `sampled`는 현재 확률 정책에서 action을 sampling한 평가입니다. 학습 초반에는 entropy가 높으므로 random baseline과 비교할 때 `sampled`도 함께 확인합니다.

```bash
python -m src.train_acac \
  --episodes 200 \
  --resume outputs/acac_p2e2/latest.pt
```

## Collaboration Flow

1. 작업 전 repo를 최신 상태로 맞춥니다.
2. `conda activate rl-team`으로 가상환경을 활성화합니다.
3. 구현/테스트 후 필요한 파일만 commit합니다.
4. `.venv`, generated trace, training output은 commit하지 않습니다.

## TODO

- [x] entropy를 학습 진행에 따라 줄여가며(annealing) 학습 — `--entropy-coef`(시작) → `--entropy-coef-final`을 `--entropy-anneal-episodes`에 걸쳐 선형 감쇠 (후반 policy drift 완화)
- [x] mini-batch + replay buffer 추가 — `--num-minibatches`(매크로 인터벌 단위 minibatch SGD), `--replay-capacity`(최근 롤아웃 재사용; PPO clip이 staleness 보정). 둘 다 기본값은 기존 동작 유지
- [x] no-op 선택 횟수 로그 추가 — 콘솔 `noop N (frac)` + `metrics.jsonl`의 `noop_decisions`/`noop_fraction`
- [x] transition 수 출력 — 콘솔 `trans T/J` (transition 수 / joint interval 수), `metrics.jsonl`엔 기존부터 기록됨

## notes
Main experiment v0(현재 버전):
  Global ready queue
  목적: heterogeneous core-task assignment 정책 검증

Extension / ablation v1:
  Per-core local queue + work stealing
  목적: 더 현실적인 OS scheduling 구조에서 정책이 유지되는지 검증
