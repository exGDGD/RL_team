# RL_team 코드베이스 개괄

> 목적: repo를 처음 읽는 사람이 "어떤 파일이 무엇을 담당하는지"와 "환경-롤아웃-학습이 어떤 순서로 이어지는지" 빠르게 파악하기 위한 구현 중심 안내서다. 연구 배경과 실험 설계의 긴 맥락은 [`teamplo-design-doc.md`](teamplo-design-doc.md), preemption 상세 설계는 [`preemption-design.md`](preemption-design.md)를 참고한다.

## 1. 한 줄 요약

이 repo는 이종 CPU 코어(P/E/Prime/LP-E)가 섞인 시스템에서 task를 어떤 코어에 배정할지 학습하는 event-driven multi-agent RL 실험 코드다.

- 1 core = 1 agent.
- task는 CPU burst와 I/O wait를 가진 multi-phase job이다.
- 환경은 SimPy 기반 discrete-event simulator다.
- RL은 ACAC 스타일의 agent-centric actor-critic 구조를 사용한다.
- 현재 학습은 단일 구성(P2E2) sanity training 중심이며, `--train-scenarios`로 여러 workload scenario를 round-robin으로 섞을 수 있다.
- 최근 패치로 `LATENCY_FLOW` reward, 병렬 rollout, GPU batched forward, baseline eval cache, preemption count logging이 추가됐다.

## 2. 최상위 구조

```text
src/
  env/                 # simulator, core/task/workload/reward/metrics
  baselines/           # random, round-robin, SJF-like, EAS-like
  rl/                  # obs batch, rollout, buffer, networks, trainer
  evaluate_baselines.py
  train_acac.py
  train_sjf_imitation.py
  train_logging.py

tests/                 # env/baseline/RL/training helper tests
docs/                  # 설계 문서와 코드베이스 안내
RL_team_testing.ipynb  # Colab/통합 테스트 노트북
```

가장 중요한 진입점은 다음 세 개다.

- `src/env/scheduler_env.py`: 스케줄링 simulator 본체.
- `src/rl/rollout.py`: env와 policy를 상호작용시켜 transition 수집.
- `src/train_acac.py`: rollout 수집, PPO/ACAC update, 평가, checkpoint 저장.

## 3. 환경 계층: `src/env/`

### `core.py`

코어 타입과 스펙을 정의한다.

- `CoreType`: `PRIME`, `P`, `E`, `LP_E`
- `CoreSpec`: speed, power, context-switch cost
- `Core`: 현재 task, busy 여부, 누적 energy 등 runtime state

실행시간은 대략 `task.current_cpu_burst / core.speed * mismatch_penalty`다. mismatch penalty는 `SchedulerEnv._mismatch_penalty()`에 있다.

### `task.py`

스케줄링 대상 task 모델이다.

- `LatencyClass`: `BEST_EFFORT`, `SOFT_RT`, `HARD_RT`
- `Task`: arrival time, CPU intensity, latency class, `cpu_bursts`, `io_waits`, progress, cost 누적, preemption 수

Preemption이 발생하면 현재 CPU burst의 일부 work만 처리하고, 잔여 burst를 줄인 뒤 task를 ready queue에 다시 넣는다.

### `workload.py`

Episode마다 stochastic workload trace를 만든다.

- `WorkloadScenario`: balanced, UI-heavy, burst-stress 등
- `WorkloadGenerator`: arrival, CPU burst, I/O wait, latency class를 샘플링

현재는 generator 기반이고, 고정 trace replay는 아직 구현되지 않았다.

### `spaces.py`

Gymnasium observation/action space의 shape 계약을 정의한다.

- `self`: 8
- `ready_queue`: task당 4
- `ready_mask`: queue size
- `other_cores`: 다른 core당 3
- `system`: 6
- `action`: `Discrete(queue_size + 1)`

`ready_queue`에는 정확한 미래 CPU burst 길이를 넣지 않는다. actor는 wait/progress/latency/intensity만 보고 추정해야 한다.

### `scheduler_env.py`

환경의 중심이다. `reset()`과 `step(actions)`를 제공하고, PettingZoo ParallelEnv와 비슷한 dict 기반 입출력을 쓴다.

주요 책임:

- task arrival / I/O completion / CPU burst finish event 진행
- ready queue 관리
- idle core dispatch
- busy core preemption eligibility 판단
- partial burst 처리
- context-switch cost 적용
- reward와 metrics 계산
- agent별 observation/action mask 생성
- 아무 dispatch 없이 future event가 사라진 idle episode가 clock을 멈춘 채 `max_env_steps`까지 도는 일을 막기 위해 `max_sim_time`까지 진행

현재 `action_mask` 의미:

- idle core가 ready task를 볼 때: action `0`은 NO-OP/대기, `1..K`는 dispatch 후보
- busy core가 preemption-eligible일 때: action `0`은 keep, `1..K`는 preempt-and-switch
- busy core가 preemption-eligible이 아닐 때: 전부 invalid, 즉 decision 대상이 아님

Action `0`은 context-dependent하다. idle core에서는 "지금 배정하지 않음", busy core에서는 "현재 task 유지"다.

## 4. Reward Mode

현재 `RewardMode`는 네 가지다.

| mode | 의미 |
|---|---|
| `event_shaped` | burst마다 work progress + energy/starvation cost, 완료 시 completion/latency 항 |
| `event_cost` | burst마다 cost만, 완료 시 completion/latency 항 |
| `completion_only` | burst 비용을 task에 누적했다가 완료 시 한 번에 reward 방출 |
| `latency_flow` | 도착 후 미완료 task마다 시간당 priority-weighted flow-time penalty 부여 |

`LATENCY_FLOW`는 최근 학습이 움직이기 시작한 핵심 변경이다. 이전 cost-only 계열 reward에서는 dispatch하지 않는 정책이 energy/runtime cost를 거의 만들지 않아 0점 근처의 나쁘지 않은 정책처럼 보일 수 있었다. `LATENCY_FLOW`는 도착했지만 끝나지 않은 task가 매 시간 penalty를 만들게 하므로 idle/no-dispatch 전략이 즉시 나빠진다.

구체적으로 `_flow_time_penalty(dt)`는 아직 완료되지 않은 in-system task마다 latency class weight를 더하고, 첫 실행 전 task에는 `response_weight`를 추가로 곱한 뒤 elapsed time `dt`만큼 penalty를 부과한다. 이 penalty는 agent별 step reward에도 반영되고, synthetic `finished_runs` event로도 추가되어 pending actor transition의 credit에 도달한다.

관련 CLI:

```bash
python -m src.train_acac \
  --reward-mode latency_flow \
  --lambda-flow 1.0 \
  --response-weight 1.5
```

## 5. Baseline 계층: `src/baselines/`

### `policies.py`

휴리스틱 policy들을 정의한다.

- `RandomPolicy`: valid ready slot 중 무작위 선택
- `RoundRobinPolicy`: ready slot 순환 선택
- `MLFQPolicy`: task별 CPU service history와 aging을 쓰는 multi-level feedback queue 휴리스틱
- `SJFLikePolicy`: 해당 core에서 runtime이 가장 짧은 ready task 선택
- `EASLikePolicy`: latency, CPU intensity, core affinity, wait bonus를 섞은 휴리스틱

### MLFQ baseline의 성격

`MLFQPolicy`는 RL actor와 observation을 맞춘 fair baseline이 아니라, task `pid`와 runtime history를 쓰는 OS-style reference baseline이다. 새 task는 높은 priority queue에서 시작하고, 누적 CPU service가 `quanta=(4.0, 12.0)` 경계를 넘으면 낮은 level로 demote된다. ready queue에서 오래 기다린 task는 `aging_threshold=30.0` 단위로 promotion된다.

현재 simulator에는 timer tick decision point가 없으므로, 전통적 MLFQ처럼 quantum 만료 즉시 preempt하지는 못한다. 대신 env가 arrival/I/O/preemption gate 등으로 scheduling decision을 열 때마다 현재 누적 service와 waiting time을 기준으로 priority를 다시 계산한다.

### SJF-like는 진짜 SJF인가?

엄밀히 말하면 "SJF-like oracle/reference"에 가깝다.

코드는 idle core마다 다음을 고른다.

```python
best_action = min(
    available,
    key=lambda action: env._runtime_on_core(core, env.ready_queue[action - 1]),
)
```

`env._runtime_on_core(core, task)`는 다음을 쓴다.

```python
task.current_cpu_burst / core.spec.speed * mismatch_penalty
```

따라서 SJF-like는 단순히 ready queue의 짧은 burst를 고르는 SJF가 아니라, "이 core에서 실행했을 때의 예상 runtime"이 가장 짧은 task를 고른다. core speed와 mismatch penalty까지 반영하므로 hetero-core-aware shortest-runtime heuristic이다.

중요한 차이:

- actor observation에는 `current_cpu_burst`가 없다.
- SJF-like는 env 내부의 `task.current_cpu_burst`와 `_runtime_on_core()`를 직접 본다.
- baseline policy들은 idle core에 대해서만 action을 낸다. 기본 학습 env는 preemption이 켜져 있으므로 RL은 preemption action surface를 갖지만 baseline은 non-preemptive로 남는다.

즉 SJF-like는 "동일 관측 조건의 공정한 SJF agent"가 아니라, simulator 내부 정보를 쓰는 강한 greedy reference다. 최종 비교에서는 `--disable-preemption` ablation이나 preemption-aware baseline을 별도로 두는 편이 더 공정하다.

## 6. RL 데이터 표현: `src/rl/`

### `obs.py`

환경 observation dict를 neural network가 쓰기 쉬운 `AgentBatch`로 바꾼다.

`AgentBatch`는 다음을 정렬된 numpy array로 들고 있다.

- agent ids
- core type indices
- self features
- ready queue features/mask
- other cores features/mask
- system features
- action mask
- decision mask
- delta t

`decision_mask`는 "이 agent가 이번 step에서 실제 결정을 해야 하는가"를 나타낸다. 현재는 action mask의 task action 중 하나라도 valid하면 decision agent로 본다.

### `buffer.py`

Rollout transition들을 담는다.

- `AgentTransition`: actor update용 agent-centric transition
- `JointMacroTransition`: critic/GAE용 shared macro timeline transition
- `RolloutBuffer`: 여러 episode buffer를 seed order로 merge하고, conflicts/invalid/preemptions/decision stats를 누적

`compute_time_scaled_gae`는 `gamma ** delta_t`를 사용해 simulated wall-clock time 길이에 따른 GAE를 계산한다.

### `rollout.py`

환경과 policy를 실제로 상호작용시키는 수집기다.

흐름:

1. `env.reset()`
2. observation을 `AgentBatch`로 변환
3. `policy.act(batch)` 호출
4. env에 action 전달
5. `env.step(actions)`로 다음 event까지 simulated time 진행
6. finished run / preemption / idle NO-OP / joint interval transition 기록
7. episode 종료까지 반복

현재 action `0` 처리:

- idle NO-OP은 `PendingDecision`으로 열리고, 다음 decision/event에서 transition으로 닫힌다.
- busy keep은 기존 running transition에 흡수된다. 별도 keep transition으로 기록하지 않는다.
- preempt-and-switch는 기존 running transition을 닫은 뒤 새 pending decision을 연다.
- `info["preemptions"]`는 `RolloutBuffer.preemptions`에 누적된다.
- `LATENCY_FLOW`의 synthetic reward event도 `finished_runs` 경로로 들어와 pending transition reward에 반영된다.

### `networks.py`

PyTorch neural network 정의다.

- `TypeSharedActor`: 코어 타입별 공유 actor. `noop_head`와 task slot head를 가진다.
- `AgentCentricCritic`: attention 기반 centralized critic. agent별 value를 출력한다.

Actor와 critic 모두 `self`, `system`, `ready_queue`, `other_cores`를 인코딩한다. 단일 코어에서 `other_cores`가 비어도 env가 `(0, 3)` shape를 보장한다.

### `trainer.py`

ACAC/PPO update의 핵심이다.

주요 단계:

1. rollout에서 actor transition과 joint transition 분리
2. critic으로 old value / next value 계산
3. joint macro timeline에서 time-scaled GAE 계산
4. actor transition이 시작된 joint index의 advantage를 actor에 매핑
5. PPO ratio / clipped objective로 actor update
6. joint return으로 critic update
7. loss, entropy, KL, clip fraction, grad norm 등 통계 반환

최근 최적화:

- `policy.act()`는 decision row를 core type별로 묶어 actor forward를 batch 처리한다.
- `evaluate_transitions()`도 transition별 1회 forward가 아니라 core type별 batched forward를 사용한다.
- critic value 계산은 agent 수가 같은 `AgentBatch`들을 묶어 한 번에 처리한다.
- claimed slot 처리는 forward 이후 sequential mask 단계로 남겨 기존 conflict-avoidance semantics를 유지한다.

현재 구현은 공식 ACAC의 GRU history encoder, target critic, PopArt 같은 안정화 장치까지 포함하지는 않는다.

### `imitation.py`

SJF-like policy의 선택을 label로 모아 actor imitation pretraining에 사용한다. PPO가 아예 움직이지 않을 때 actor 표현력과 action-mask 경로를 sanity check하는 용도에 가깝다.

## 7. 학습 스크립트

### `train_acac.py`

메인 학습 entrypoint다.

하는 일:

- CLI argument 파싱
- `SchedulerEnv` 생성
- `TorchACACPolicy`와 `ACACTrainer` 생성
- 여러 episode rollout 수집
- trainer update
- deterministic/sample eval
- random/MLFQ/SJF/EAS baseline 평가
- `metrics.jsonl`, `train.log`, `latest.pt`, `best.pt` 저장

기본 device는 `auto`다. CUDA가 있으면 CUDA, 없으면 CPU를 쓴다. CUDA 사용 시 TF32 matmul/conv를 켜서 Colab T4/Ampere 계열에서 MLP/attention forward를 빠르게 한다.

병렬 rollout:

```bash
python -m src.train_acac \
  --device auto \
  --rollout-workers 4 \
  --rollout-episodes 32
```

Worker process는 CPU policy snapshot으로 environment rollout만 수집하고, main process의 GPU policy는 PPO update에 사용된다. CUDA process fork 문제를 피하기 위해 `spawn` start method를 쓴다. 병렬 worker 결과는 seed 순서로 merge해 sequential collection과 episode id / joint index가 같도록 유지한다.

Baseline 평가는 학습 policy에 의존하지 않으므로 같은 eval seed/config에서는 매번 같은 값이다. `cached_evaluate_baselines()`가 이를 캐시해서 eval 때마다 random/MLFQ/SJF/EAS episode를 다시 도는 비용을 줄인다.

중요한 로그:

- `reward`: training rollout의 환경 총 reward 평균
- `eval reward`: deterministic argmax 평가
- `sampled`: 현재 stochastic policy sampling 평가
- `base`: random/MLFQ/SJF/EAS baseline
- `eval/<scenario>`: scenario별 deterministic/sampled reward와 baseline reward
- `entropy`: action distribution entropy
- `clip`: PPO clipping 비율
- `preempt`: rollout/eval에서 발생한 preemption 횟수
- `choices`, `forced`: 실제 배울 수 있는 decision이 얼마나 있었는지

### 왜 최근 패치 후 학습이 움직였나

가장 큰 요인은 `LATENCY_FLOW`다. 학습 신호가 "어떤 선택이 turnaround/response를 줄이는가"에 더 직접적으로 연결됐다. 이전 reward에서는 모든 task가 결국 완료되는 trace에서 progress/completion 합이 episode마다 비슷하거나, 반대로 dispatch하지 않는 행동이 비용을 덜 만들어 좋아 보이는 문제가 있었다. `LATENCY_FLOW`는 미완료 task가 있는 동안 시간이 흐를수록 penalty가 쌓이므로, 빠르게 dispatch하고 완료시키는 정책이 일관되게 더 좋은 return을 받는다.

보조 요인:

- 병렬 rollout으로 같은 wall-clock 시간에 더 많은 episode를 본다.
- batched actor/critic forward로 GPU 사용 효율이 올라간다.
- baseline cache로 eval 비용이 크게 줄어 training loop가 덜 멈춘다.
- idle/no-dispatch episode가 `max_env_steps=10000`까지 헛도는 병목이 줄었다.
- preemption count 로그로 정책이 실제로 preemption을 쓰는지 확인 가능해졌다.

정리하면 reward mode가 학습 방향을 바꾼 핵심이고, 나머지는 속도와 관측 가능성을 높인 요인이다.

학습/평가 distribution은 `train_acac.py`의 scenario 옵션으로 조절한다. 기본값은 모두 `all`이라 네 workload scenario가 episode 순서에 따라 round-robin 배정된다.

- `--train-scenarios`: training rollout distribution
- `--eval-scenarios`: 학습 중 checkpoint 선택용 validation distribution (`--eval-seed`, 기본 10000)
- `--test-scenarios`: 학습 종료 후 best checkpoint에 대해 한 번 실행하는 held-out test distribution (`--test-seed`, 기본 20000)

예를 들어 `--train-scenarios all --rollout-episodes 16`은 네 workload scenario를 한 update에 4개씩 섞는다. 기존 single-scenario sanity run을 재현하려면 `--train-scenarios balanced`를 명시한다. 각 update의 mix는 `metrics.jsonl`의 `train_scenarios` 필드에 남고, validation/test mix는 evaluation summary의 `scenario_counts`에 남는다.

### `train_sjf_imitation.py`

SJF-like label로 actor를 지도학습한 뒤 checkpoint를 저장한다.

```bash
python -m src.train_sjf_imitation --device auto --output outputs/sjf_imitation/actors.pt
```

저장된 actor는 PPO warm-start에 사용할 수 있다.

```bash
python -m src.train_acac \
  --pretrained-actors outputs/sjf_imitation/actors.pt \
  --output-dir outputs/acac_sjf_warm_start
```

### `train_logging.py`

콘솔과 `train.log`에 사람이 읽기 좋은 한 줄 로그를 남긴다. 구조화된 전체 기록은 `metrics.jsonl`에 저장된다.

## 8. 테스트 구조

테스트는 기능별로 나뉜다.

- `test_scheduler_env.py`: env reset/step/reward/preemption semantics
- `test_metrics.py`: episode metric 계산
- `test_baselines.py`: baseline runner/policy 동작
- `test_rl_obs.py`: observation batch shape/action mask
- `test_rl_buffer.py`: GAE와 rollout buffer
- `test_rl_rollout.py`: async rollout, pending decision, preempt transition, idle policy spin guard
- `test_rl_networks.py`: actor/critic forward pass
- `test_rl_trainer.py`: PPO/ACAC update step
- `test_rl_imitation.py`: SJF imitation dataset
- `test_train_acac.py`: training helper, checkpoint/version/device helper
- `test_parallel_rollout.py`: rollout seed ordering, process-boundary pickle safety
- `test_reward_latency_flow.py`: `LATENCY_FLOW` reward properties

로컬 환경에 torch가 없으면 torch 기반 테스트는 skip될 수 있다. Colab이나 torch 설치 환경에서는 actor/critic forward와 trainer update까지 전체 테스트를 돌려야 한다.

## 9. 한 episode의 데이터 흐름

```text
WorkloadGenerator
  -> SchedulerEnv.reset()
  -> observations dict
  -> build_agent_batch()
  -> policy.act(batch)
  -> SchedulerEnv.step(actions)
  -> finished_runs / rewards / next observations
  -> RolloutBuffer.append_joint()
  -> AgentTransition append/close
  -> ACACTrainer.update()
  -> checkpoint/log/eval
```

이 흐름에서 자주 헷갈리는 점은 env step과 actor transition이 1:1이 아니라는 것이다.

- env step은 simulator가 다음 scheduling-relevant event까지 진행한 구간이다.
- actor transition은 어떤 agent의 dispatch/preempt/NO-OP decision이 credit을 받는 구간이다.
- critic은 shared joint macro interval 위에서 value와 advantage를 계산한다.

## 10. 현재 구현상 주의점

1. SJF-like baseline은 actor보다 더 많은 내부 정보를 쓴다. "SJF보다 못함"은 강한 oracle-like reference보다 못하다는 뜻이지, 동일 관측 조건에서 진짜 SJF보다 못하다는 뜻은 아니다.
2. Baseline policy들은 idle core에만 action을 낸다. Preemption-enabled RL과 비교할 때 action surface가 다르다.
3. Idle NO-OP은 현재 action mask상 가능하다. `LATENCY_FLOW`를 쓰면 ready task를 방치하는 시간이 바로 flow-time penalty가 된다.
4. `ready_queue`에는 정확한 future burst 길이가 없다. actor는 latency, intensity, wait, progress만 보고 추정해야 한다.
5. 학습은 아직 단일 core configuration(P2E2) sanity 단계다. workload scenario는 `--train-scenarios`로 섞을 수 있지만, core-config domain randomization, curriculum, fixed trace eval은 남아 있다.
6. PPO 로그에서 `policy_loss`가 0에 가까운 것은 그 자체로 비정상은 아니다. advantage normalization 후 첫 epoch ratio가 1 근처면 loss scalar는 작게 보일 수 있다. 대신 `kl`, `ratio_std`, `actor_grad`, eval action 분포를 함께 봐야 한다.
7. Reward mode를 바꾸면 checkpoint를 이어받기보다 새 output dir에서 시작하는 편이 해석이 깔끔하다. 특히 `LATENCY_FLOW`는 reward scale/분포가 기존 event reward와 다르다.

## 11. 추천 읽기 순서

처음부터 코드를 읽는다면 아래 순서가 덜 헷갈린다.

1. `src/env/task.py`
2. `src/env/core.py`
3. `src/env/scheduler_env.py`
4. `src/rl/obs.py`
5. `src/rl/rollout.py`
6. `src/rl/buffer.py`
7. `src/rl/networks.py`
8. `src/rl/trainer.py`
9. `src/train_acac.py`
10. `tests/test_reward_latency_flow.py`
11. `tests/test_parallel_rollout.py`

환경 semantics를 바꾸는 작업은 대개 `scheduler_env.py`, `obs.py`, `rollout.py`, 관련 tests를 함께 봐야 한다. 학습 안정화 작업은 `trainer.py`, `train_acac.py`, `train_logging.py`, Colab 로그를 함께 보는 편이 좋다.
