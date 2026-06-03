# 이종(Heterogeneous) CPU 스케줄링을 위한 비동기 MARL 실험 설계 문서

> **프로젝트:** ACAC 알고리즘 기반 Heterogeneous CPU Scheduling
> **목적:** 팀원이 본 문서만으로 실험의 동기, 환경 설계, 검증 방법을 이해하도록 한다.
> **상태:** ~~설계 단계 (구현 시작 전)~~ → **구현 진행 중** — 환경/Baseline/단일 구성(P2E2) ACAC sanity 학습 + **선점(preemption)/NO-OP 활성/burst 비노출**까지 구현 (W5–W6 + 6/3 패치). 🔧
> **기준 커밋:** `dev` `ac22ca6` (6/3 패치 item 1 burst 제거, 2026-06-03). *(원 설계 기준은 `0366c5d`, 2026-05-31)*
> **git:** https://github.com/exGDGD/RL_team/tree/dev
> **drive:** https://drive.google.com/drive/folders/1pwQYCoJ-rBIghPgGU1bfnvwb2HkzcbYX

**마커 범례:** 🔧 [구현 반영] 설계안과 다르게 구현됨 · ✅ [해소/완료] 과거 열린 쟁점이 해결됨 · ⚠️ [확인 필요/미구현] · 🆕 [신규] 설계안에 없던 추가.

---

## 이 문서의 변경 요약 (5/15 설계안 → 현재 구현)

> 아래 표는 **원본 설계 문서 대비 코드에서 실제로 달라진 부분**만 모은 것입니다. 본문 각 절에도 같은 내용을 마커로 표시했습니다. **6/3 패치(preemption / NO-OP / burst 비노출)** 로 과거 ⚠️ 항목 다수가 ✅ 로 바뀌었습니다.

| 영역 | 설계안 | 현재 구현 | 마커 |
|---|---|---|---|
| 문서 상태 | 구현 전 | sanity 학습 + 선점/NO-OP/burst 비노출 | 🔧 |
| 코어 스펙 | 4종 (수치) | **동일**. mismatch penalty를 구체 규칙으로 구현 | 🔧 |
| 워크로드 축 | CPU/병렬성/지연 (병렬성 보류) | 병렬성 축 **제거 확정**, multi-phase(burst+IO) 모델 구현 | 🔧 |
| Task 구조 | 3축 좌표 | `cpu_intensity, latency_class, cpu_bursts[], io_waits[]` | 🔧 |
| Observation: ready queue | task 3축 + 대기시간만 | `[waiting, cpu_progress, latency, intensity]` **4-dim, 정확 burst 비노출** | ✅ |
| Observation: other cores | 진행 task 3축 포함 | 진행 task 정보 **제외** (타입/busy/경과만) | 🔧 |
| Observation: self | 코어타입 one-hot + 직전 task 타입 | 코어타입 **인덱스 스칼라** + busy/경과/에너지/Δt + **현재 실행 task latency/intensity/progress (8-dim)** | 🔧 |
| Action | NO-OP 포함 | 구현됨. 학습 시 **NO-OP 활성(allow_noop=True)**, 학습 transition으로 기록 | ✅ |
| Preemption | 5/19에 추가 논의 | **구현됨** (M2 interruptible-options, P1 우선순위/P3 기아 게이트 + min_run, 부분 burst) | ✅ |
| Reward 모드 | dense+sparse 병행 | **3 모드** (event_shaped / event_cost / completion_only) | 🔧 |
| Starvation 항 | $\sum W_k^2 \cdot \Delta t$ | **log 기반** `(mean(log1p W)+0.5·max(log1p W))·Δt` | 🔧 |
| Context-switch 페널티 | $\lambda_C=1.0$ 적용 | **preempt switch 시 적용** (코어 타입별 cs 비용 × λ_C) | ✅ |
| GAE | $\gamma^{\Delta t}$ time-scaled | 구현됨 + **joint macro-timeline** critic 추가 | 🔧 |
| 같은타입 coordination | 순차 결정(obs 반영) | **사후 conflict 해소**(먼저인 코어 우선, 중복=NO-OP) | 🔧 |
| Domain randomization/curriculum | 핵심 실험 | **미구현** (단일 P2E2 고정) | ⚠️ |
| SJF imitation 사전학습 | (계획에 없음) | **신규 추가** (actor를 SJF로 warm-start) | 🆕 |
| 평가 trace replay | balanced_v1~10 등 고정 trace | stochastic generator만, **replay 미구현** | ⚠️ |
| Baseline | 7종 | Random/RoundRobin/SJF/EAS **4종 구현**, 나머지 미구현 | 🔧 |

---

## 0. 한눈에 보기 (TL;DR)

- **문제:** P-Core, E-Core 등 성능이 다른 CPU 코어가 혼합된 환경에서, 어떤 태스크를 어느 코어에 보낼지 결정하는 스케줄러를 강화학습으로 학습한다.
- **알고리즘:** ACAC (Agent-Centric Actor-Critic for Asynchronous MARL, ICML 2025). 각 태스크의 실행 시간이 다른 비동기 환경에 적합.
- **에이전트:** 1 core = 1 agent. 코어 타입별로 정책 파라미터 공유 (최대 4개의 actor).
- **핵심 실험 목표:** **하나의 학습된 정책이 다양한 코어 구성(예: P2E2, P1E3, P3E1, ...)에 zero-shot으로 작동**한다는 것을 보인다. (Scalability claim)
- **방법:** Domain randomization으로 코어 구성과 워크로드를 매 에피소드 무작위 sampling.
- **검증:** 12개 코어 구성 × 5개 워크로드 시나리오 grid에서 baseline 대비 성능 측정.

> 🔧 **[구현 반영]** 현재 코드는 위 비전 중 **환경/보상/단일 구성 학습 파이프라인 + 선점(preemption)/NO-OP 학습/burst 비노출 관측**까지 구현된 상태이며, **domain randomization·scalability 평가는 아직 미착수**다. 즉 TL;DR은 최종 목표를 기술하며, 실제 진척은 §11 마일스톤을 참고.

---

## 1. 연구 배경과 동기

### 1.1 왜 CPU 스케줄링인가

현대 SoC(Apple Silicon, Intel Lunar Lake, Snapdragon Elite 등)는 단일 종류의 코어가 아닌, **성능–효율 비대칭 코어**를 혼합 탑재한다. OS 스케줄러는 다음을 동시에 만족해야 한다.

- **Throughput** (처리량) 최대화
- **Energy** (전력 소모) 최소화
- **Responsiveness** (특히 UI 이벤트의 지연 최소화)
- **Starvation** (기아 현상) 방지

기존의 휴리스틱 스케줄러(CFS, EAS, WLF)는 수동 튜닝된 가중치로 이 다목적 최적화를 처리한다. 본 연구는 **각 코어를 독립 에이전트로 보고, 다중 에이전트 강화학습으로 스케줄링 정책을 학습**한다.

### 1.2 왜 ACAC인가 — 핵심 정당화

CPU 스케줄링의 본질적 어려움은 **비동기성**이다. 각 태스크의 burst time이 다르므로, 일반적인 동기 MARL(MAPPO, QMIX)을 그대로 적용하면 다음 문제가 발생한다.

- **Zero-padding 문제:** 짧은 태스크가 끝나도 다른 에이전트가 끝날 때까지 대기를 강제하면 의사결정 시점이 왜곡됨.
- **시간 차원 정렬 문제:** wall-clock과 step index의 불일치로 GAE 계산이 부정확해짐.

ACAC는 세 가지로 이를 해결한다.

| ACAC 특성 | 본 환경에서의 의미 | 구현 상태 |
|---|---|---|
| 에이전트별 독립 시간 흐름 (Δt를 observation에 포함) | 각 코어가 자기 결정 간격을 인지 | ✅ `self`에 `Δt_since` 포함 |
| Attention 기반 Critic | 다른 코어의 실시간 상태를 동적으로 집계 | ✅ `AgentCentricCritic` (MultiheadAttention, 4 heads) |
| 시간 스케일링 할인율 γ^Δt | 짧은/긴 태스크 간 보상 가치의 일관성 확보 | ✅ `compute_time_scaled_gae` |

> 🔧 **[구현 반영]** Critic은 단순히 에이전트별 가치를 내는 데 더해, **공유 macro-timeline(JointMacroTransition) 위에서 advantage를 계산**하도록 구현됐다(`compute_joint_advantages`). 비동기 에이전트들의 보상을 wall-clock 기준으로 정렬해 critic을 학습시키는 구조로, 설계안에는 없던 추가 메커니즘이다.

특히 attention critic은 **가변 개수의 코어**를 자연스럽게 처리할 수 있다 — 이 점이 본 연구의 scalability 실험에서 결정적으로 활용된다.

---

## 2. 환경의 4-Tier 코어 아키텍처

환경은 4종류의 이종 CPU 코어로 구성된다. 각 코어의 스펙은 다음과 같다. **(코드 `src/env/core.py`의 `CORE_SPECS`와 일치)**

| Core Type | 처리 속도 배율 | 전력 계수 | Context-Switch 비용 | 주 용도 |
|---|---:|---:|---:|---|
| **Prime-Core** | 4.0× | 8.0 | 1.5 ms | 단일 스레드 최고 성능 |
| **P-Core** | 3.0× | 5.0 | 1.0 ms | 헤비 병렬 연산 |
| **E-Core** | 1.5× | 1.5 | 0.3 ms | I/O, 인터럽트 |
| **LP-E Core** | 0.8× | 0.4 | 0.2 ms | 백그라운드 장기 태스크 |

처리 속도 배율은 base burst time을 나누는 값으로 정의: 실제 실행시간 $\Delta t = B_{\text{base}} / s_i \cdot \alpha_{\text{mismatch}}$. 여기서 $\alpha_{\text{mismatch}} \geq 1$은 부적합 매칭 페널티.

> 🔧 **[구현 반영]** $\alpha_{\text{mismatch}}$가 추상 개념에서 **구체 규칙**으로 구현됨 (`SchedulerEnv._mismatch_penalty`):
> - HARD-RT 태스크를 E / LP-E에 배정 → **×1.4**
> - `cpu_intensity > 0.75` 태스크를 LP-E에 배정 → **×1.5**
> - `cpu_intensity < 0.25` 태스크를 Prime / P에 배정 → **×1.15**
> - 그 외 → ×1.0
>
> 🔧 **[구현 반영]** 기본 코어 구성은 **P2E2** (`DEFAULT_CORE_CONFIG = {P:2, E:2}`). Prime/LP-E는 스펙은 정의돼 있으나 현재 학습/테스트 기본값에는 미포함.
>
> 🔧 **[구현 반영]** Context-Switch 비용은 이제 **선점(preemption) 시 실제 reward 페널티로 사용**된다(§5.3). 전환을 유발한 코어에 `λ_C × cs_cost` 를 청구한다.

> *수치는 초기 가정값이며 실험 중 튜닝 대상이다.*

---

## 3. 워크로드 모델

### 3.1 왜 단순한 "5종 분류"가 아닌가

초기 안에서는 워크로드를 5종(Single-Thread Burst / Heavy Parallel / Massive I/O / Background / UI)으로 분류했으나, 검토 결과 분류 축 불일치·관측 불가능·경계 케이스 표현 불가 문제가 있어 폐기했다.

### 3.2 직교 축 표현 (채택) — 병렬성 축 제거

> 🔧 **[구현 반영]** 설계안의 3축 중 **병렬성(Parallelism) 축은 제거 확정**됐다(코드에 존재하지 않음). 대신 태스크가 **multi-phase 구조**(CPU burst와 I/O wait의 반복)를 갖도록 구현됐다.

현재 Task가 실제로 보유하는 정보 (`src/env/task.py`):

```
cpu_intensity   : 한 quantum 평균 CPU 사용률 [0, 1]
latency_class   : BEST_EFFORT(0) / SOFT_RT(1) / HARD_RT(2)
cpu_bursts[]    : CPU를 점유하는 구간들의 시간 (phase별)
io_waits[]      : 각 burst 사이 I/O 대기 시간 (len = phase-1)
```

즉 한 태스크는 `burst → io → burst → io → ... → burst` 형태로 진행되며, 매 CPU burst가 끝날 때마다 새 스케줄링 결정이 발생한다(5/19 회의 결정 반영). 선점이 켜지면 burst 중간에도 결정점이 생길 수 있다(§5.2).

#### Task 생성 방식
- **Agent(core)에게 노출되는 정보:** `waiting_time, cpu_progress, latency_class, cpu_intensity` — ✅ **정확한 burst 수치(`current_cpu_burst`, `remaining_cpu_work`)는 노출하지 않는다** (6/3 패치 item 1, 아래 §5.1 Layer 2 참고).
- **Task가 내부적으로 가진 정보:** cpu_bursts, io_waits (phase 횟수/시간), latency_class, pid. (시뮬레이터 내부 계산·SJF/EAS baseline 라벨에만 사용)

### 3.3 5종은 클러스터(mode)로 해석

기존 5종 라벨은 별도 카테고리가 아니라 이 공간에서 자주 나타나는 mode로 본다. 현재 `WorkloadScenario`로 구현된 것:

| 시나리오 (코드 enum) | 특성 (latency 분포 / cpu_intensity 분포) |
|---|---|
| `BALANCED` | latency [0.45, 0.35, 0.20], beta(2,2) |
| `UI_HEAVY` | latency [0.15, 0.35, 0.50] (지연 민감 多), 낮은 cpu |
| `BG_HEAVY` | latency [0.75, 0.20, 0.05] (best-effort 多) |
| `BURST_STRESS` | 높은 cpu_intensity(beta(5,1.8)), burst_scale 2배 |

> 🔧 **[구현 반영]** 설계안의 5종 라벨 대신 위 **4개 시나리오 enum**으로 구현됨. 시나리오는 reward의 라벨 oracle이 아니라 **태스크 생성 분포**를 결정한다.

### 3.4 단순화 가정 (명시)

- 태스크 간 의존성 없음 (lock/barrier 없음).
- 캐시 친화도는 context-switch 비용으로만 표현.
- (병렬성 미모델링 — §3.2)

---

## 4. 워크로드 생성

### 4.1 현재 구현: Stochastic Generator

> 🔧 **[구현 반영]** `WorkloadGenerator` (`src/env/workload.py`)가 구현됨:
> - 도착: 지수분포(평균 `1/arrival_rate`)로 sampling, `episode_time` 또는 `max_tasks`(기본 64)까지.
> - cpu_intensity: 시나리오별 Beta 분포.
> - latency_class: 시나리오별 categorical.
> - phase 수: `{1,2,3}` (확률 `[0.6, 0.3, 0.1]`), burst는 Gamma, io_wait는 Gamma.

### 4.2 Stochastic / Replay 분리 — 일부만 구현

| 모드 | 용도 | 동작 | 상태 |
|---|---|---|---|
| **Stochastic** | 학습 | 매 에피소드 분포에서 sampling (seed로 재현) | ✅ 구현 |
| **Replay** | 평가 | 미리 sampling한 고정 trace 재생 | ⚠️ **미구현** |

> ⚠️ **[확인 필요 / TODO]** 현재는 `seed`를 고정하면 generator가 동일 trace를 재생성하므로 재현성은 확보되지만, 설계안의 **별도 trace 파일 세트**(`balanced_v1~v10`, `ui_heavy`, `borg_subset`, `adversarial`)와 trace 직렬화/replay 로더는 아직 없다. 평가 grid 실행 전 구현 필요(마일스톤 W4).

---

## 5. State / Action / Reward 정의

### 5.1 Observation 구조 — 구현 기준

> 🔧 **[구현 반영]** 실제 observation은 `src/env/spaces.py`와 `SchedulerEnv._observe_agent`에 정의돼 있으며, 설계안과 차원·필드가 다소 다르다. 아래는 **현재 코드 기준** 정의다.

#### Layer 1: Self (`self`, 8-dim) — `SELF_FEATURE_DIM = 8`
```
[ core_type_index,            # 0..3 (one-hot 아님, 스칼라 인덱스)   🔧 변경
  busy (0/1),                 # 🔧 추가
  elapsed_current,            # 현재 task 진행 경과
  accumulated_energy,         # 누적 에너지
  Δt_since,                   # 직전 결정 이후 경과 시간
  running_latency_class,      # 🔧 현재 실행 중 task latency (idle=0)  ← 선점 판단용
  running_cpu_intensity,      # 🔧 현재 실행 중 task intensity (idle=0)
  running_cpu_progress ]      # 🔧 현재 실행 중 task progress (idle=0)
```
> 🔧 **[변경]** 설계안의 "코어 타입 원-핫(4-dim)"은 **인덱스 스칼라 1개**로, "직전 처리 태스크 타입"은 **제거**됐다. 6/3 패치로 **현재 실행 중 task의 latency/intensity/progress 3칸이 추가**돼 8-dim이 됐다 — busy 코어가 "현재 task를 유지할지 선점할지"를 판단하려면 자기 task를 봐야 하기 때문. (정확 burst는 여기서도 노출하지 않음.)

#### Layer 2: Ready Queue (`ready_queue`, K×4) — `READY_TASK_FEATURE_DIM = 4`, `K=8`
```
슬롯별:
[ waiting_time,           # 누적 대기 W_k
  cpu_progress,           # 지금까지 처리된 CPU work
  latency_class,          # 0/1/2
  cpu_intensity ]         # [0,1]
+ ready_mask (K-bit)      # 유효 슬롯 표시
```

> ✅ **[해소 — 6/3 패치 item 1]** 과거 코드는 `current_cpu_burst`·`remaining_cpu_work`를 ready queue에 **직접 노출**해, SJF 류 최적 결정을 거의 "정답지"로 주는 셈이었다(5/19 회의의 "burst 정보 비노출" 결정과 상충). **이제 두 필드를 제거**(6→4 차원)해, agent는 정확한 미래 burst를 보지 못하고 `waiting/progress/latency/intensity`로 추정해야 한다. 정확 burst는 시뮬레이터 내부와 SJF/EAS baseline 라벨에서만 쓰인다. burst를 가린 뒤의 "복구 메커니즘"으로 선점(§5.2)을 함께 도입했다.

#### Layer 3: Other Cores (`other_cores`, (N-1)×3) — `OTHER_CORE_FEATURE_DIM = 3`
```
다른 코어별:
[ core_type_index,
  busy (0/1),
  elapsed ]               # 진행 중 task 경과 시간
```
> 🔧 **[변경]** 설계안의 "진행 중 태스크 3축 수치"는 **포함하지 않는다**. 타입/busy/경과 시간만 제공하며, critic의 attention이 이 시퀀스를 집계한다.

#### Layer 4: System Context (`system`, 6-dim) — `SYSTEM_FEATURE_DIM = 2 + 4`
```
[ num_cores (N),
  mean_utilization,
  type_counts × 4 ]       # 타입별 코어 수
```
> 🔧 **[변경]** 설계안의 "**내 타입의 다른 코어 수**" 전용 필드는 별도로 두지 않고, `type_counts`와 `self`의 `core_type_index`에서 파생 가능하도록 했다. Scalability 학습 시 이 표현으로 충분한지는 ablation(Layer 4 제거)으로 검증 예정.

추가로 `action_mask` (K+1 bit)가 obs에 포함된다. 선점이 켜지면 env가 이 마스크로 결정 자격을 인코딩한다(비적격 busy 코어 = 전부 0).

### 5.2 Action Space

`a_i ∈ {0, 1, …, K}` (`build_action_space = Discrete(K+1)`):
- `a_i = 0`: **변화 없음** — idle 코어면 NO-OP(의도적 idle), busy 코어면 **현재 task 유지(keep)**.
- `a_i ∈ {1,…,K}`: ready queue의 `a_i`번째 태스크 선택. busy 코어가 고르면 현재 task를 **선점(preempt)** 하고 그 슬롯으로 전환.

**Invalid action masking 적용** — 빈 슬롯/이미 점유된 task는 mask 처리.

> ✅ **[해소 — 6/3 패치 item 2]** 과거에는 학습 시 `allow_noop=False`로 NO-OP을 비활성화하고, rollout에서 NO-OP 결정을 버렸다. **이제 `allow_noop=True`(기본)** 이며, idle 코어의 NO-OP(대기)와 busy 코어의 keep을 **학습 transition으로 기록**한다(`rollout.py`). 즉 "언제 쉴지/유지할지"도 학습 대상. 무한루프는 wakeup 게이트 + `_advance_to_decision`의 force-progress로 방지.

> ✅ **[해소 — 6/3 패치 item 3] Preemption 구현.** 5/19에 합의했던 선점이 구현됐다. 모델/게이트:
> - **결정 모델 (M2, interruptible options):** event-driven 비동기를 유지하되, **wakeup(태스크 도착·I/O 완료) 시점**에 아래 게이트를 통과한 busy 코어도 결정점을 받는다. transition은 "run이 끝날 때"가 아니라 "run이 **멈출 때(완료 OR 선점)**" 닫힌다.
> - **check_preempt 게이트 (env=적격성, policy=선택):**
>   - **P1 우선순위:** ready task의 `latency_class > 실행 중 task의 latency_class` (HARD-RT가 BEST-EFFORT를 밀어냄). ↔ 실제 OS의 RT-preempts-CFS.
>   - **P3 기아:** ready task의 `waiting_time > starvation_threshold`(기본 100). ↔ 공정성/anti-starvation.
>   - **min_run 가드:** 방금 시작한 task(`elapsed < preempt_min_run`)는 선점 후보에서 제외(thrashing 방지). ↔ CFS min_granularity.
>   - *(EAS misfit 류 P2 affinity 이주는 이번 범위에서 제외.)*
> - **부분 burst:** 선점 시 실행한 만큼만 진행에 반영하고, 남은 burst를 ready queue에 재삽입(`Task.preempt_current_burst`).
> - **비용:** 전환을 유발한 코어에 컨텍스트 스위치 비용 청구(§5.3, §14-B 해소).
> - **플래그:** `enable_preemption`(학습 기본 on). 비선점 ablation은 `train_acac --disable-preemption`. action 인코딩은 통합 K+1(action space·actor head 무변경).

### 5.3 Reward 함수 — 3개 모드로 일반화

> 🔧 **[구현 반영]** 설계안의 "dense + sparse 병행"이 **3개의 명시적 reward mode**로 구현됨 (`RewardMode`):
>
> | 모드 | 설명 |
> |---|---|
> | `EVENT_SHAPED` | 매 burst마다 work shaping + 비용, 완료 시 completion/latency 항 (dense) |
> | `EVENT_COST` | burst 시 work shaping 없이 비용만, 완료 시 completion/latency |
> | `COMPLETION_ONLY` | burst 비용은 내부 누적, 완료 시 한 번에 모두 지급 (sparse) |

**현재 `RewardWeights` 기본값** (`src/env/scheduler_env.py`):

| 항목 | 값 |
|---|---|
| `progress_work` | 1.0 |
| `completion` | 5.0 |
| `completion_work` | 5.0 |
| `energy` (λ_E) | 0.1 |
| `starvation` (λ_S) | 0.1 |
| `latency` (λ_L) | 0.05 |
| `context_switch` (λ_C) | 1.0 |
| `work_norm` | 10.0 |
| `starvation_max_wait_weight` | 0.5 |

**1) CPU burst 종료 시 (`EVENT_SHAPED` 기준):**
$$R_i = \underbrace{w_p \cdot \tfrac{\text{work}}{\text{work\_norm}}}_{\text{progress}} - \lambda_E R_{\text{energy}} - \lambda_S R_{\text{starv}} \;(+\; \text{완료 시 completion 항})$$

- **work:** 처리한 CPU work를 `work_norm`으로 정규화.
- **energy:** $p_i \cdot \Delta t$ (전력계수 × 실행시간).
- **starvation:** 🔧 **[변경]** 설계안의 $\sum W_k^2 \cdot \Delta t$ (제곱)이 **log 기반으로 교체**됨 (commit `reward 개선(starvation 항 정규화)`):
  $$R_{\text{starv}} = \big(\text{mean}_k \log(1+W_k) + 0.5 \cdot \max_k \log(1+W_k)\big)\cdot \Delta t$$
  제곱항의 발산 위험을 없애면서 장기 대기 태스크에 가중(max 항)을 유지.

**2) 태스크 완료 시:**
$$R_i \mathrel{+}= \text{completion} + w_{cw}\cdot\tfrac{\text{total\_work}}{\text{work\_norm}} - \lambda_L \cdot (\text{latency\_class} \times \text{turnaround})$$

- **latency:** 지연 민감도(`latency_class`) × **turnaround time**으로 페널티. (commit `response time to throughput time for latency criteria`에서 기준 시간 정리.)

**3) 선점 시 (preempt):**
- 선점된 task는 부분 burst만큼의 progress/energy/starvation 비용을 정산하고(완료 항 없음), 남은 work를 큐로 되돌린다.
- ✅ **[해소 — Context-switch 페널티 적용]** 전환을 유발한 코어에 $\lambda_C \times \text{cs\_cost}(\text{core\_type})$ 를 reward 페널티로 청구한다(`_resolve_and_dispatch`). 과거 "가중치만 정의·미적용"(§14-B) 상태가 해소됐다.

> 🔧 **[구현 반영]** 추가 정규화 2종:
> - **Team reward:** 각 step 보상을 에이전트 수로 나눠 공유(commit `team reward`).
> - **`reward_scale`** (trainer 기본 0.01): advantage 계산 전 보상 스케일 다운으로 value 분산 완화.

### 5.4 시간 스케일 GAE

ACAC의 핵심: 정수 step이 아닌 wall-clock Δt 기반 할인 (`compute_time_scaled_gae`).
$$\delta_i^{(t)} = R_i^{(t)} + \gamma^{\Delta t_i^{(t)}} V(s_{t+1}) - V(s_t)$$

> 🔧 **[구현 반영]** advantage는 에이전트별 transition이 아니라 **joint macro-timeline**(`JointMacroTransition`) 위에서 계산된 뒤, 각 에이전트 transition이 자신의 `joint_index`로 해당 advantage를 참조한다(`compute_joint_advantages`). 비동기 보상을 공통 시간축으로 묶기 위한 구조.

---

## 6. 에이전트 정의

### 6.1 1 Core = 1 Agent
각 코어가 독립적으로 자기 ready queue에서 태스크를 선택하는 의사결정자. ✅ 구현.

### 6.2 정책 공유 구조: Type Sharing
**Type sharing 채택** — 타입별 actor (`TypeSharedActor`). 같은 타입 코어는 같은 파라미터를 공유하므로 코어 수가 바뀌어도 네트워크 재사용 가능(scalability). ✅ 구현.

### 6.3 Actor / Critic 구조
- **Actor:** `TypeSharedActor` (타입별, decentralized). self/system/other를 인코딩해 context를 만들고, NO-OP head와 task head로 logit 산출, action_mask 적용. ✅
- **Critic:** `AgentCentricCritic` (centralized, `nn.MultiheadAttention` 4 heads). 다른 코어들을 attention으로 집계해 에이전트별 value 산출. ✅
- CTDE 패러다임. ✅

### 6.4 같은 타입 코어 간 coordination
> 🔧 **[변경]** 설계안은 "순차 결정(앞 코어 결정을 다음 코어 obs에 반영) + stochastic policy"였으나, 현재 구현은 **사후 conflict 해소** 방식이다 (commit `core간 conflict 처리`):
> - 모든 에이전트가 동시에 행동을 내고, `_resolve_and_dispatch`에서 **결정적 에이전트 순서상 먼저인 코어가 task를 차지**한다.
> - 같은 task를 고른 뒤순위 코어의 행동은 **NO-OP으로 처리**되고, rollout에서 해당 pending decision은 폐기(`_discard_rejected_decisions`)된다.
> - conflict / invalid_action 통계는 buffer에 집계되어 모니터링된다.
>
> stochastic policy(entropy)로 자연 다양화하는 부분은 유지. 순차 obs 반영은 미구현. *(선점이 켜지면 busy 코어도 결정점에 참여하므로 같은 conflict 해소 로직이 선점-전환에도 적용된다.)*

---

## 7. 핵심 실험: Domain Randomization for Scalability

> ⚠️ **[미구현 — 다음 단계]** 본 절의 내용(코어 수/타입비율/도착률/믹스 randomization, 3단계 curriculum, 병렬 rollout)은 **아직 구현되지 않았다.** 현재 `train_acac.py`는 *"Single-config ACAC sanity training"*으로, **P2E2 고정** 구성에서 학습/평가만 수행한다(선점은 이 구성에서도 동작). 아래는 목표 설계로 유지하며, 마일스톤 W7부터 착수.

### 7.1 무엇을 보이려고 하는가
> 한 번 학습된 ACAC 정책이 다양한 코어 구성에 zero-shot으로 잘 작동한다. (같은 칩셋 패밀리 변종에 단일 정책 배포)

### 7.2 무엇이 랜덤화되는가 (목표)
| 차원 | 범위 | 비고 |
|---|---|---|
| 총 코어 수 $N$ | Uniform{4,…,12} | scalability 핵심 변수 |
| 타입 비율 | Dirichlet | 극단 구성 포함 |
| 도착률 $\lambda$ | LogUniform[0.05, 0.5] | 워크로드 강도 |
| 태스크 mix | Dirichlet | 분포 변화 |

코어 구성과 워크로드 분포는 **독립 sampling**.

### 7.3 Curriculum Learning (목표)
| Stage | 비중 | 범위 |
|---|---|---|
| Stage 1 (Narrow) | ~20% | N=4, P2E2 위주 |
| Stage 2 (Medium) | ~40% | N∈{4,6,8} |
| Stage 3 (Full) | ~40% | 전체 분포 |

### 7.4 학습 시 주의사항 (목표/일부 반영)
| 항목 | 대응 | 상태 |
|---|---|---|
| Reward scale 환경 간 차이 | PopArt/running-stats normalization | ⚠️ 미구현 (현재 `reward_scale` 상수만) |
| Value function 분산 | PPO value clipping | ✅ clip_ratio·grad clip 존재 |
| 표본 효율 | 16~32 env 병렬 rollout | ⚠️ 현재 순차(`rollout-episodes` 단순 반복) |
| 학습 곡선 noise | 5 seeds, CI 보고 | ⚠️ 평가 단계 |
| LP-E 결정 빈도 보정 | 타입별 transition 가중 | ⚠️ 미구현 |

---

## 8. 평가 프로토콜

> ⚠️ **[부분 구현]** 평가 메트릭 계산(`src/env/metrics.py`)과 baseline 단일 에피소드 실행(`evaluate_baselines.py`, `baselines/runner.py`)은 구현됐으나, **12×5 grid 전체 실행 / per-config 히트맵 / Pareto front 분석은 미구현**이다. *(baseline 정책은 idle 코어에만 행동하므로 선점-환경에서도 절대 선점하지 않는다 — 비선점 baseline vs 선점 RL 의 공정 비교가 된다.)*

### 8.1 평가 Grid (목표)
코어 구성 ~12 × 워크로드 시나리오 ~5 × seed 20 → 약 1,200 에피소드. 평가 병렬화 필요.

### 8.2 핵심 분석 지표 (목표)
1. Per-configuration breakdown (scalability 핵심 증거)
2. Train vs Test distribution shift
3. OOD extrapolation (N=16,20)
4. Generalist vs Specialist gap (90%+ 이면 강한 결과)

### 8.3 평가 메트릭 — 구현됨
> 🔧 **[구현 반영]** `EpisodeMetrics`로 다음이 계산된다:
> - throughput (tasks/time), total_energy
> - mean / p95 / **p99** response time, mean / p95 / p99 turnaround time
> - mean / p95 ready_wait time
> - **starvation_rate** (max_ready_wait > threshold(기본 100) 비율)
> - mean / per-core utilization
>
> Pareto Front(Throughput–Energy–Latency 3축) 시각화는 분석 스크립트 수준에서 추가 예정.

---

## 9. Baseline 비교군

> 🔧 **[구현 반영]** 현재 구현된 baseline (`src/baselines/policies.py`):

| Baseline | 구현 | 비고 |
|---|---|---|
| **Random** | ✅ | (설계안에 없던 sanity용 추가) |
| **Round-Robin** | ✅ | |
| **Shortest Job First (SJF-like)** | ✅ | `_runtime_on_core` 최소 task 선택 |
| **EAS-like Heuristic** | ✅ | affinity + energy bias + latency + wait bonus 점수화 |
| Specialist ACAC (per-config) | ⚠️ 미구현 | Generalist upper bound |
| Generalist MAPPO (zero-padding) | ⚠️ 미구현 | attention 우위 검증 |
| Generalist ACAC w/o Layer 4 obs | ⚠️ 미구현 | ablation |
| Generalist ACAC w/ fixed-step GAE | ⚠️ 미구현 | ablation |

> 🆕 **[신규 — 학습 안정화]** **SJF Imitation 사전학습**이 추가됐다(`src/rl/imitation.py`, `train_sjf_imitation.py`, commit `imitate-sjf test`). SJF baseline의 결정을 라벨로 actor를 지도학습 warm-start한 뒤 ACAC로 미세조정한다. 이는 baseline이라기보다 **학습 파이프라인의 초기화 전략**이며, 설계안에는 없던 항목이다. (학습이 진행되지 않는 시나리오 B에 대한 선제 대응 성격. burst 비노출 이후엔 4-dim obs로 SJF를 *추정* 학습하는 형태가 된다.)

> 🔧 **[ablation 가능]** `--disable-preemption` 으로 비선점 ACAC를 학습할 수 있어, **선점 유/무 비교**가 새 ablation 축으로 추가됐다.

---

## 10. 시나리오별 위험 분석과 대응

(설계안 유지 — 아래는 현재 구현 관점의 보강만 표시)

### 시나리오 A — Generalist가 specialist에 크게 뒤짐
대응: Layer 4 obs 확장 → Hierarchical policy → MoE. *(domain randomization 구현 후 검증 가능)*

### 시나리오 B — 학습 자체가 진행 안 됨
> 🔧 **[선제 대응 반영]** SJF imitation warm-start, reward 정규화(starvation log화, team reward, reward_scale)가 이미 이 위험을 겨냥해 도입됨. *(NO-OP은 비활성→활성으로 바뀌었으므로, 학습이 NO-OP에 갇히는지 모니터링 필요.)* 추가로 curriculum Stage 1 비중 확대 예정.

### 시나리오 C — OOD(N=16,20) 성능 폭락
대응: 솔직 보고, 학습 범위 내 generalization만 claim.

### 시나리오 D — ACAC ≈ MAPPO
대응: 비대칭 구성 + burst workload 셀 강조 분석.

---

## 11. 단계별 마일스톤 — 진척 반영

> 🔧 **[구현 반영]** 커밋 로그(`48ab639`~`ac22ca6`, 2026-05-18 ~ 06-03) 기준 현재 진척:

| 주차 | 목표 | 상태 |
|---|---|---|
| W1–W2 | 환경 골격 (이산 이벤트 시뮬레이터, core/task 클래스) | ✅ 완료 (SimPy 기반, Global Queue) |
| W3 | Reward 단위 테스트, 휴리스틱 baseline 3종 | ✅ 완료 (+Random 포함 4종, tests/ 다수) |
| W4 | Stochastic generator + replay + 평가 trace 세트 | 🔧 **부분** — generator 완료, **replay/trace 세트 미구현** |
| W5–W6 | Single-config ACAC 학습 (P2E2 sanity) | 🔧 **진행 중** — rollout/trainer/GAE/imitation 구현, 학습 튜닝 단계 |
| (6/3 패치) | **선점(preemption) + NO-OP 활성 + burst 비노출** | ✅ **완료** — env+RL 활성화, 테스트 통과 |
| W7 | Domain randomization 인프라 (병렬 env, curriculum) | ⬜ 미착수 |
| W8–W10 | Domain randomization 학습 + ablation | ⬜ 미착수 |
| W11 | 평가 grid 전체 실행, 분석 | ⬜ 미착수 |
| W12 | 결과 정리, 리포트 | ⬜ 미착수 |

> 🆕 계획 외 추가 작업: SJF imitation 사전학습, 3종 reward mode, joint macro-timeline critic, team reward/정규화, **선점·부분 burst·CS 비용·NO-OP 학습**.

---

## 12. 본 연구의 기여 요약
(설계안 유지 — 최종 목표 기준)
1. 이종 CPU 스케줄링을 비동기 MARL 문제로 정형화.
2. (병렬성 제외) CPU 집약도·지연 민감도·multi-phase burst 기반 워크로드 모델링.
3. Domain randomization + attention critic으로 scalable 정책 학습.
4. Multi-objective reward 설계 (log 기반 starvation 페널티, 선점 시 context-switch 비용 포함).
5. 같은 칩셋 패밀리 변종에 단일 모델 배포 가능성 검증.

---

## 13. 용어 정리
(변경 없음 — 원본 참조)

| 용어 | 의미 |
|---|---|
| MARL | Multi-Agent Reinforcement Learning |
| ACAC | Agent-Centric Actor-Critic |
| CTDE | Centralized Training, Decentralized Execution |
| GAE | Generalized Advantage Estimation |
| PPO | Proximal Policy Optimization |
| Domain Randomization | 환경 파라미터 무작위화 학습 |
| Zero-shot transfer | 미경험 셋업에 재학습 없이 적용 |
| OOD | Out-of-Distribution |
| Burst time | 태스크가 CPU를 점유하는 시간 |
| Starvation | 큐 장기 대기 기아 현상 |
| Preemption | 실행 중 태스크를 중단하고 다른 태스크로 전환 |

---

## 14. 열린 질문 (Open Questions) — 갱신

6/3 패치로 **A·B·C가 해소**됐고, D와 기존 질문이 남는다.

**해소됨 (6/3 패치)**
- ~~**A. Agent observation에 `current_cpu_burst` / `remaining_cpu_work`를 노출할 것인가?**~~ → ✅ **비노출로 결정·구현**(item 1). ready queue obs 4-dim, 정확 burst 제거.
- ~~**B. Context-switch 페널티를 reward에 실제 반영할 위치/방식.**~~ → ✅ **선점 전환 시 적용**(전환 유발 코어에 `λ_C × cs_cost`).
- ~~**C. Preemption action을 언제 도입할지.**~~ → ✅ **구현**(M2 interruptible-options, P1/P3 게이트, `enable_preemption` 기본 on).

**남은 질문**
- **D. Replay trace 세트 포맷·생성 시점.** (평가 grid 전 필수 — §4.2) ⚠️ 미해결.
- (선점 후속) min_run / starvation_threshold 등 게이트 하이퍼파라미터 튜닝, EAS misfit(P2) affinity 이주 추가 여부.

**기존 질문(유지)**
1. Real-world trace(Google Borg)를 1차에 포함할지.
2. DVFS를 행동 공간에 포함할지.
3. Multi-phase task를 1차/ablation 중 어디에. *(→ 현재 multi-phase는 이미 1차 환경에 구현됨; ablation 대상은 phase 수/IO 모델)*
4. Reward 가중치를 학습 가능 hyper-param으로 둘지.
5. Specialist baseline을 12개 전부/대표 4~5개 중 어디까지 학습할지.

---

## 부록. 회의 기록 (원본 보존)

### 5/19 회의
- Task 생성 시 설정 값: 끝날 때까지 걸리는 시간, CPU 직접 활용 시간, I/O 대기 시간/횟수, 지연민감도, PID → ✅ 구현(`cpu_bursts`, `io_waits`, `latency_class`, `pid`).
- Agent observation: 자기 정보, 큐 상황, 큐 task의 대기시간/cpu 경과/지연민감도. **핵심: burst 정보는 agent에게 주지 않고 자율 학습** → ✅ **구현(6/3 패치 item 1으로 burst 비노출 적용).**
- 다른 agent 정보(작업중/경과) obs 포함 → ✅ 구현(`other_cores`).
- Preemption action → ✅ **구현(6/3 패치 item 3).**
- I/O queue 구현 → ✅ 구현(`_release_io`, `io_waits`).
- 지연민감도 reward 직접 반영 → ✅ 구현(latency × turnaround).

### 05/22
- environment setting (Global Queue), reward mode 도입, latency 기준을 response→turnaround로 정리 (commit `1086657`, `77e9829`).

### 06/03 (패치)
- **item 1** ready queue obs에서 `current_cpu_burst`/`remaining_cpu_work` 제거(6→4) → SJF 정답지 비노출.
- **item 2** NO-OP 활성(`allow_noop=True`) + 학습 transition 기록.
- **item 3** preemption 구현(M2, wakeup P1/P3+min_run 게이트, 부분 burst, CS 비용). self obs 5→8(현재 task 노출).

*문서 끝. (기준 커밋 `ac22ca6`, 2026-06-03)*
