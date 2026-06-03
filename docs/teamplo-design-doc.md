# 이종(Heterogeneous) CPU 스케줄링을 위한 비동기 MARL 실험 설계 문서

> **출처:** teamplo 팀 문서 (강화학습개론 Final · 실험 환경 계획)
> https://www.teamplo.com/team/hbrhMrf8HcHM/jilGtKO7UsVY
> **GitHub:** https://github.com/exGDGD/RL_team/tree/dev
> **프로젝트:** ACAC 알고리즘 기반 Heterogeneous CPU Scheduling
> **목적:** 팀원이 본 문서만으로 실험의 동기, 환경 설계, 검증 방법을 이해하도록 한다.
> **상태:** 구현 진행 중 — 환경/Baseline/단일 구성(P2E2) ACAC sanity 학습까지 구현 (대략 W5–W6).
> **기준 커밋:** `dev` 브랜치 `0366c5d` (imitate-sjf test, 2026-05-31)

이 파일은 teamplo 외부 설계 문서를 저장소 지식그래프에 연결하기 위해 보존한 사본이다(원본 보존, 마커 유지).

## 변경 요약 (5/15 설계안 → 현재 구현)

| 영역 | 설계안 | 현재 구현 | 마커 |
|---|---|---|---|
| 문서 상태 | 구현 전 | 단일 구성 sanity 학습 단계 | 🔧 |
| 코어 스펙 | 4종 (수치) | 동일. mismatch penalty를 구체 규칙으로 구현 | 🔧 |
| 워크로드 축 | CPU/병렬성/지연 | 병렬성 축 제거 확정, multi-phase(burst+IO) 모델 구현 | 🔧 |
| Task 구조 | 3축 좌표 | `cpu_intensity, latency_class, cpu_bursts[], io_waits[]` | 🔧 |
| Observation: ready queue | task 3축 + 대기시간만 | `current_cpu_burst`, `remaining_cpu_work`까지 노출 | ⚠️ |
| Observation: other cores | 진행 task 3축 포함 | 진행 task 정보 제외 (타입/busy/경과만) | 🔧 |
| Observation: self | 코어타입 one-hot + 직전 task 타입 | 코어타입 인덱스 스칼라, 직전 task 타입 제거 | 🔧 |
| Action | NO-OP 포함 | 구현됨. 단 학습 시 NO-OP 비활성(allow_noop=False) | 🔧 |
| Preemption | 5/19에 추가 논의 | 미구현 (burst 끝까지 실행) | ⚠️ |
| Reward 모드 | dense+sparse 병행 | 3 모드 (event_shaped / event_cost / completion_only) | 🔧 |
| Starvation 항 | ∑W²·Δt | log 기반 `(mean(log1p W)+0.5·max(log1p W))·Δt` | 🔧 |
| Context-switch 페널티 | λ_C=1.0 적용 | 가중치 정의됨, reward에 아직 미적용 | ⚠️ |
| GAE | γ^Δt time-scaled | 구현됨 + joint macro-timeline critic 추가 | 🔧 |
| 같은타입 coordination | 순차 결정(obs 반영) | 사후 conflict 해소(먼저인 코어 우선, 중복=NO-OP) | 🔧 |
| Domain randomization/curriculum | 핵심 실험 | 미구현 (단일 P2E2 고정) | ⚠️ |
| SJF imitation 사전학습 | (계획에 없음) | 신규 추가 (actor를 SJF로 warm-start) | 🆕 |
| 평가 trace replay | balanced_v1~10 등 고정 trace | stochastic generator만, replay 미구현 | ⚠️ |
| Baseline | 7종 | Random/RoundRobin/SJF/EAS 4종 구현 | 🔧 |

## 0. TL;DR
- **문제:** P-Core/E-Core 등 성능이 다른 CPU 코어가 혼합된 환경에서 어떤 태스크를 어느 코어에 보낼지 결정하는 스케줄러를 RL로 학습.
- **알고리즘:** ACAC (Agent-Centric Actor-Critic for Asynchronous MARL, ICML 2025).
- **에이전트:** 1 core = 1 agent. 코어 타입별 정책 파라미터 공유 (최대 4 actor).
- **핵심 목표:** 하나의 학습 정책이 다양한 코어 구성(P2E2, P1E3, P3E1…)에 zero-shot 작동(Scalability claim).
- **방법:** Domain randomization으로 코어 구성·워크로드를 매 에피소드 무작위 sampling.
- **검증:** 12 코어 구성 × 5 워크로드 grid에서 baseline 대비 성능 측정.

## 1. 연구 배경과 동기
현대 SoC(Apple Silicon, Intel Lunar Lake, Snapdragon Elite)는 성능–효율 비대칭 코어를 혼합 탑재. OS 스케줄러는 Throughput 최대화 / Energy 최소화 / Responsiveness / Starvation 방지를 동시에 만족해야 함. 본 연구는 각 코어를 독립 에이전트로 보고 MARL로 스케줄링 정책을 학습.

**왜 ACAC인가:** CPU 스케줄링의 본질적 어려움은 비동기성. 동기 MARL(MAPPO/QMIX)은 zero-padding 문제와 시간 차원 정렬 문제 발생. ACAC는 (1) 에이전트별 독립 시간 흐름(Δt를 obs에 포함), (2) attention 기반 critic, (3) γ^Δt 시간 스케일링 할인으로 해결. attention critic은 가변 코어 수를 자연스럽게 처리 → scalability 실험에 결정적. 추가로 공유 macro-timeline(JointMacroTransition) 위에서 advantage 계산(compute_joint_advantages).

## 2. 4-Tier 코어 아키텍처 (src/env/core.py CORE_SPECS)
| Core Type | 처리속도 배율 | 전력 계수 | Context-Switch 비용 | 주 용도 |
|---|---|---|---|---|
| Prime-Core | 4.0× | 8.0 | 1.5 ms | 단일 스레드 최고 성능 |
| P-Core | 3.0× | 5.0 | 1.0 ms | 헤비 병렬 연산 |
| E-Core | 1.5× | 1.5 | 0.3 ms | I/O, 인터럽트 |
| LP-E Core | 0.8× | 0.4 | 0.2 ms | 백그라운드 장기 태스크 |

실행시간 Δt = B_base / s_i · α_mismatch. mismatch penalty 규칙(SchedulerEnv._mismatch_penalty):
- HARD-RT 태스크를 E/LP-E에 배정 → ×1.4
- cpu_intensity > 0.75 태스크를 LP-E에 → ×1.5
- cpu_intensity < 0.25 태스크를 Prime/P에 → ×1.15
- 그 외 → ×1.0

기본 코어 구성: **P2E2** (DEFAULT_CORE_CONFIG = {P:2, E:2}). Prime/LP-E는 스펙만 정의, 기본 학습/테스트엔 미포함.

## 3. 워크로드 모델
병렬성 축 제거 확정. 태스크는 multi-phase 구조(CPU burst ↔ I/O wait 반복). Task 정보(src/env/task.py): `cpu_intensity`, `latency_class`(BEST_EFFORT/SOFT_RT/HARD_RT), `cpu_bursts[]`, `io_waits[]`. 매 CPU burst 종료 시 새 스케줄링 결정 발생.

시나리오(WorkloadScenario enum): BALANCED / UI_HEAVY / BG_HEAVY / BURST_STRESS — reward의 oracle이 아니라 태스크 생성 분포를 결정.

## 4. 워크로드 생성
WorkloadGenerator(src/env/workload.py): 도착=지수분포, cpu_intensity=시나리오별 Beta, latency=categorical, phase 수 {1,2,3}=[0.6,0.3,0.1], burst/io=Gamma. Stochastic(학습, seed 재현)은 구현, Replay(평가용 고정 trace)는 미구현(W4 TODO).

## 5. State / Action / Reward (구현 기준)
**Layer 1 Self (5-dim):** [core_type_index(스칼라), busy, elapsed_current, accumulated_energy, Δt_since]
**Layer 2 Ready Queue (K×6, K=8):** [waiting_time, cpu_progress, latency_class, cpu_intensity, **current_cpu_burst**, **remaining_cpu_work**] + ready_mask
> ⚠️ §14-A: 5/19 회의는 "burst 정보 비노출, 자율학습" 결정. 그러나 코드는 current_cpu_burst·remaining_cpu_work를 직접 노출 → SJF 정답지화 우려, 재확정 필요.

**Layer 3 Other Cores ((N-1)×3):** [core_type_index, busy, elapsed] (진행 task 수치 제외)
**Layer 4 System (2+4):** [num_cores, mean_utilization, type_counts×4] + action_mask

**Action:** a_i ∈ {0,…,K}. a_i=0은 NO-OP. 학습 시 allow_noop=False. Preemption 미구현(non-preemptive).

**Reward 3 모드(RewardMode):** EVENT_SHAPED / EVENT_COST / COMPLETION_ONLY.
RewardWeights 기본값: progress_work 1.0, completion 5.0, completion_work 5.0, energy(λ_E) 0.1, starvation(λ_S) 0.1, latency(λ_L) 0.05, context_switch(λ_C) 1.0, work_norm 10.0, starvation_max_wait_weight 0.5.
- burst 종료 시: progress(work/work_norm) − λ_E·energy − λ_S·starv (+완료 시 completion).
- starvation: log 기반 `(mean log(1+W) + 0.5·max log(1+W))·Δt`.
- 완료 시: completion + w_cw·total_work/work_norm − λ_L·(latency_class × turnaround).
> ⚠️ context_switch는 가중치만 정의, reward 계산식 미반영. 적용 위치/방식 결정 필요.
- 정규화: team reward(에이전트 수로 분배), reward_scale(trainer 0.01).

**시간 스케일 GAE:** δ = R + γ^Δt·V(s')−V(s). advantage는 joint macro-timeline(compute_joint_advantages)에서 계산.

## 6. 에이전트
1 Core = 1 Agent. Type sharing(TypeSharedActor). Actor: TypeSharedActor(decentralized, NO-OP head + task head + action_mask). Critic: AgentCentricCritic(centralized, nn.MultiheadAttention 4 heads). CTDE. 같은타입 coordination은 사후 conflict 해소(_resolve_and_dispatch: 먼저인 코어 우선, 중복=NO-OP, _discard_rejected_decisions).

## 7. 핵심 실험: Domain Randomization for Scalability (⚠️ 미구현)
목표: 한 번 학습된 ACAC 정책이 다양한 코어 구성에 zero-shot 작동. 랜덤화: 총 코어 수 N~Uniform{4..12}, 타입 비율 Dirichlet, 도착률 LogUniform[0.05,0.5], task mix Dirichlet. 3단계 curriculum(Narrow/Medium/Full). 현재 train_acac.py는 P2E2 고정 single-config sanity training.

## 8. 평가 프로토콜 (⚠️ 부분 구현)
평가 grid: ~12 코어구성 × ~5 워크로드 × seed 20 ≈ 1,200 에피소드(미구현). 메트릭(src/env/metrics.py, EpisodeMetrics 구현): throughput, total_energy, mean/p95/p99 response·turnaround, ready_wait, starvation_rate(threshold 100), utilization. Pareto Front 시각화는 추가 예정.

## 9. Baseline (src/baselines/policies.py)
구현됨: Random / Round-Robin / SJF-like(_runtime_on_core 최소) / EAS-like(affinity+energy+latency+wait 점수화). 미구현: Specialist ACAC, Generalist MAPPO(zero-padding), ACAC w/o Layer4, ACAC w/ fixed-step GAE(ablation들).
🆕 SJF Imitation 사전학습(src/rl/imitation.py, train_sjf_imitation.py): SJF 결정을 라벨로 actor를 지도학습 warm-start 후 ACAC fine-tune. 시나리오 B(학습 안 됨) 대응.

## 10. 위험 분석
- A. Generalist≪Specialist → Layer4 확장 → Hierarchical → MoE.
- B. 학습 안 됨 → 🆕 SJF imitation warm-start, reward 정규화, NO-OP 비활성 이미 도입.
- C. OOD(N=16,20) 폭락 → 솔직 보고, 학습 범위 내 generalization만 claim.
- D. ACAC≈MAPPO → 비대칭 구성 + burst workload 강조.

## 11. 마일스톤 진척 (48ab639~0366c5d, 2026-05-18~05-31)
- W1–W2 환경 골격(SimPy, Global Queue) ✅
- W3 reward 단위테스트 + 휴리스틱 baseline ✅ (+Random 4종)
- W4 stochastic generator ✅ / replay·trace 세트 🔶 미구현
- W5–W6 single-config ACAC(P2E2) 🔶 진행 중 (rollout/trainer/GAE/imitation 구현, 튜닝 단계)
- W7 domain randomization 인프라 ⬜
- W8–W10 DR 학습 + ablation ⬜ / W11 평가 grid ⬜ / W12 리포트 ⬜
🆕 계획 외: SJF imitation, 3 reward mode, joint macro-timeline critic, team reward/정규화.

## 14. 열린 질문 (우선 결정)
- **A.** obs에 current_cpu_burst/remaining_cpu_work 노출 여부 (회의 결정과 상충).
- **B.** context-switch 페널티 reward 반영 위치/방식.
- **C.** preemption action 도입 시점.
- **D.** replay trace 세트 포맷·생성 시점.
기존: Google Borg trace 1차 포함 여부, DVFS 행동공간 포함, reward 가중치 학습화, specialist baseline 범위.

## 부록. 회의 기록
**5/19:** Task 설정값(끝까지 시간/CPU 사용/IO 대기/지연민감도/PID) ✅, agent obs에 burst 비노출 결정(⚠️ 코드는 노출), 다른 agent 정보 obs 포함 ✅, preemption ⚠️ 미구현, I/O queue ✅, latency reward ✅.
**5/22:** Global Queue 환경, reward mode 도입, latency 기준 response→turnaround 정리 (commit 1086657, 77e9829).
