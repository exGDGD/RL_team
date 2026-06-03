# Preemption 설계 스펙 (6/3 패치 item 3 구체화)

> **상태:** 설계 확정 대기 (구현 전)
> **선행 논의:** teamplo 6/3 패치 — item 1(obs에서 burst 제거)·item 2(NO-OP 활성)·item 3(preemption+CS 비용)
> **결정된 순서:** preemption(+NO-OP)을 **먼저** 구현 → 그다음 obs burst 제거(item 1).
> 이유: burst를 먼저 빼면 "obs 불확실 + 복구 불가"가 겹쳐 중간 단계 에이전트가 최악이 됨. preemption이 "돌려보고 고치는" 복구 메커니즘을 먼저 제공해야 함.

---

## 1. 결정 시점 모델: M2 (interruptible options)

현재(M1)는 semi-MDP — 각 action이 "한 burst 동안 task를 돌리는" 시간확장 행동이고, **idle 코어만** run 경계에서 결정한다. 한 코어는 자기 run이 끝날 때만 마이크를 잡으므로 "하던 걸 멈춤"(preempt)을 표현할 수 없다.

**M2:** event-driven·비동기는 유지하되, preemption-relevant 이벤트가 나면 **busy 코어에게도 그 시점에 결정권**을 준다. transition 정의를 확장한다:

> transition은 run이 **끝날 때**가 아니라 run이 **멈출 때(완료 OR preempt)** 닫힌다.

preempt는 "run을 멈추는 또 다른 사유"로, 기존 `finished_runs` 기계에 이른 종료 이벤트로 얹는다. ACAC의 비동기·가변 Δt(γ^Δt) 설계를 깨지 않는다.

---

## 2. 트리거 = wakeup 이벤트

선점을 *고려*하는 시점은 실제 OS의 스케줄러 호출 지점과 1:1로 맞춘다.

| 실제 OS | 우리 env | 비고 |
|---|---|---|
| voluntary (block/exit/yield) | burst 종료(완료/I/O) | **기존 M1 결정점 그대로** |
| wakeup (task runnable) | `_release_arrivals`(도착), `_release_io`(I/O 완료) | **이미 env 안에 발생** — 지금은 idle 코어 있을 때만 멈춤 |
| timer tick | (미채택, = M3 quantum) | ACAC 비동기성과 충돌해 보류 |
| load balance/EAS | (P2, 제외) | 아래 §3 참고 |

**유일한 코드 변경 지점:** `_advance_to_decision`(`scheduler_env.py:238`)이 지금은 "ready task + idle 코어"일 때만 멈춘다. 모든 코어가 busy인데 HARD_RT가 도착하면 멈추지 않고 시간을 흘려버린다. M2는 wakeup이 아래 게이트를 통과하면 **idle 코어가 없어도** 결정점을 연다.

---

## 3. check_preempt 게이트 (env = 적격성)

실제 스케줄러는 wakeup마다 전체 재스케줄을 돌리지 않고 값싼 `check_preempt_curr`로 게이팅한다. 우리도 둘로 나눈다 — **env는 "언제 물어볼지"(적격성), policy는 "실제 바꿀지"(선택).**

wakeup 발생 시 아래 사유 중 하나라도 성립하면 해당 busy 코어에 결정점 부여:

- **P1 — 우선순위 선점:** `ready.latency_class > running.latency_class`. HARD_RT 도착이 BEST_EFFORT를 밀어냄. ↔ RT-preempts-CFS. **O(1) 정수 비교, 실제 fast-path.**
- **P3 — 기아 완화:** `ready.waiting_time > starvation_threshold`(env 기본 100). ↔ CFS fairness/anti-starvation. **가장 오래 기다린 ready task만 추적하면 O(1).**

**제외 — P2(misfit/affinity 이주):** EAS misfit은 실제 OS에서 wakeup이 아니라 주기적 load-balance(softirq)에서 처리되고, 우리 게이트의 cadence와 맞지 않아 이번 범위에서 제외.

**thrashing 가드:** 실행 task의 `elapsed_current < min_run`이면 선점 후보에서 제외(방금 시작한 건 안 건드림). ↔ CFS `min_granularity`.

---

## 4. action 인코딩 (policy = 선택) — *확정 (2026-06-03)*

**통합 K+1, 원자적 preempt+switch.** action space·actor head 무변경.

```
action space: Discrete(K+1)   # 그대로

통합 의미:  0 = no change
           j = reassign me to slot j

idle 코어:  0 = 대기(NO-OP) | j = 슬롯 j dispatch
busy 코어:  0 = 현재 task 유지(keep) | j = 현재 task PREEMPT + 슬롯 j로 전환
                                        (잔여 work 재큐잉, CS 비용 청구)

actor head: noop_head + task_head -> K+1 logits  (무변경)
```

`noop_head`가 자연스럽게 "no change(대기/유지)" 헤드가 되고, slot 헤드가 "(재)배정"이 된다. busy 여부는 `self` obs에 있으므로 type-shared actor가 조건부로 학습 가능.

> 대안(미채택): 명시적 preempt action(K+2, 2단계+슬롯 가로채기 race), 통합 K+1+합법성 하드마스크(thrashing 하드 차단·학습 자유도↓). CS 비용으로 soft 억제하는 위 추천안을 기본으로 한다.

---

## 5. 수반되는 변경

### 5a. obs — `self`에 현재 실행 task 노출 (5 → 8)

busy 코어가 keep-vs-preempt를 판단하려면 자기 task를 봐야 한다. 정확 burst(`current_cpu_burst`)는 **넣지 않는다** — item 1의 정신(정답지 비노출) 유지.

**`self` 최종 레이아웃 (8-dim):**

| idx | feature | idle일 때 | 정규화(`normalize_observation_tensors`) |
|---|---|---|---|
| 0 | core_type_index | 코어 타입 | `/ (len(CoreType)-1)` |
| 1 | busy (0/1) | 0 | 없음 |
| 2 | elapsed_current = `now - task_started_at` | 0 | `log1p` |
| 3 | accumulated_energy | 누적 | `log1p` |
| 4 | Δt_since = `now - last_decision_time` | — | `log1p` |
| 5 | running.latency_class | 0 | `/ 2.0` (ready_queue와 동일 척도) |
| 6 | running.cpu_intensity | 0 | 없음 (이미 [0,1]) |
| 7 | running.cpu_progress | 0 | `log1p` |

`SELF_FEATURE_DIM` 5→8, Box low/high도 3칸 추가(`[0,0,0]` / `[2.0, 1.0, inf]`). `networks.py`는 `SELF_FEATURE_DIM` 상수로 actor·critic 모두 자동 전파(직접 수정 없음). `trainer.normalize_observation_tensors`는 기존 `self[:, 2:] = log1p` 일괄 처리를 **인덱스별로 분리**해야 함(5번=`/2.0`, 6번=무정규화).

**전체 obs 최종 형태 (preemption 단계):**

| key | shape | 비고 |
|---|---|---|
| `self` | (8,) | ← 본 변경 |
| `ready_queue` | (K, 6) | item 1에서 (K, 4)로 축소 예정 |
| `ready_mask` | (K,) | |
| `other_cores` | (N-1, 3) | 변경 없음 |
| `system` | (6,) | 변경 없음 |
| `action_mask` | (K+1,) | 변경 없음 (통합 K+1) |

### 5b. Task — 부분 burst 추적
현재 `cpu_progress`는 `finish_current_burst`에서 **full burst**만 더한다(`task.py:79-87`). preempt는 burst 중간 종료이므로:
- 현재 burst 내 실행량(elapsed_work)을 추적.
- preempt 시 현재 burst의 잔여 = `current_cpu_burst − elapsed_work`로 줄이고 task를 ready_queue로 재삽입.

### 5c. context-switch 비용 (§14-B 해소)
switch 시 코어 타입별 CS 비용(core spec: Prime 1.5 / P 1.0 / E 0.3 / LP-E 0.2) × `RewardWeights.context_switch`(현재 1.0, 정의만 되고 미적용)를 페널티로 적용. **CS 비용은 preempt를 *유발한* action(슬롯 j 선택)에 귀속.**

### 5d. transition 기록 — "no change" 행동도 기록 (= item 2 흡수)
"keep"(busy의 0)과 "NO-OP"(idle의 0)은 같은 **no-change** 행동이다. "언제 preempt/대기할지"를 학습하려면 둘 다 transition으로 남아야 한다. 현재 `rollout.py:53-54`의 `if action == 0: continue`(NO-OP 드롭)를 수정한다. **이로써 item 2(NO-OP 활성)가 preemption 구현에 자연 포함된다.**
무한루프: wakeup 게이트가 시간 경계라, no-change가 반복돼도 다음 wakeup까지 시간이 흐른다(별도 강제 tick 불필요). 단, "ready 있고 idle 있는데 전원 NO-OP" 케이스는 다음 이벤트로 진행하도록 `_advance_to_decision`에서 보장.

---

## 6. credit / transition 역학 (preempt 발생 시)

코어 C가 task A를 실행 중, 결정점에서 policy가 슬롯 j(=task B)로 switch:
1. A의 run 조기 종료 → `finished_runs`에 이른 종료 이벤트(부분 cpu_work, 부분 energy, starvation, **completion 없음**).
2. A의 잔여 work → `ready_queue` 재삽입.
3. CS 페널티는 B 선택 action에 청구(§5c).
4. C의 "dispatch A" pending transition이 닫힘(부분 보상 귀속). 이어서 "dispatch B" 새 pending transition 시작.

---

## 7. 영향 파일 (구현 체크리스트)

| 파일 | 변경 |
|---|---|
| `src/env/scheduler_env.py` | `_advance_to_decision`(wakeup 게이트 P1/P3+min_run), `_resolve_and_dispatch`(busy 코어 처리·preempt·CS), `_dispatch`/`_complete_finished_runs`(부분 burst·이른 종료), `_observe_agent`(self 8-dim) |
| `src/env/task.py` | 부분 burst 실행량 추적·preempt 시 잔여 갱신·재큐잉 |
| `src/env/spaces.py` | `SELF_FEATURE_DIM` 5→8, self Box 갱신 |
| `src/rl/obs.py` | `decision_mask`에 "eligible busy 코어" 포함, self 파싱 |
| `src/rl/rollout.py` | no-change(0) 행동 기록(`if action==0: continue` 수정) |
| `src/rl/trainer.py` | `allow_noop` 기본값 검토, `normalize_observation_tensors` self 인덱스 조정 |
| `src/rl/networks.py` | `SELF_FEATURE_DIM`으로 자동 전파(직접 수정 없음) |
| `tests/` | `test_scheduler_env`·`test_rl_obs`·`test_rl_rollout`(noop)·`test_rl_networks` 등 다수 갱신 |

---

## 8. 검증 한계

로컬 셸에 numpy/torch/pytest 없음(`rl-team` conda env 미접근) → **학습 경로를 로컬에서 못 돌림.** `py_compile` + env 단위 추론으로만 검증 가능. 최종 검증은 `rl-team`에서 `pytest -q` 필요.

---

## 9. 이후 단계: item 1 (obs에서 burst 제거)

위 preemption이 자리 잡은 뒤 진행:
- `ready_queue` 6→4: `current_cpu_burst`·`remaining_cpu_work` 제거(`scheduler_env._observe_agent`, `spaces.READY_TASK_FEATURE_DIM`, `trainer.normalize_observation_tensors`의 4:6 log1p, `train_acac.summarize_rollout_actions`의 두 키, 관련 테스트).
- 이 시점엔 preemption이 복구 메커니즘을 제공하므로 "정답지 제거"가 안전.
