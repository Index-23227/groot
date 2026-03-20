# Task Roadmap — 단계별 완성도 업그레이드

> **Task**: 물체를 집어서 지정 위치에 놓기 (Pick & Place)
> **Robot**: Doosan E0509 (6-DOF + Gripper)
> **Camera**: Global view 1대 (wrist cam 없음)
> **핵심 원칙**: 데모는 한 번만 수집. 시스템 완성도를 소프트웨어로 올린다.

---

## 전체 구조

```
                    ┌─────────────────────────────────────────────┐
                    │              시스템 아키텍처                  │
                    └─────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ 입력     │    │ 두뇌         │    │ 안전 장치     │    │ 출력     │
  │          │ ──▶│              │ ──▶│              │ ──▶│          │
  │ L0: 하드코딩│    │ GR00T N1.6   │    │ ActionAdapter│    │ E0509    │
  │ L2: STT  │    │ SmolVLA      │    │ Temporal     │    │ servoj   │
  │ L3: 대화형│    │              │    │ Blender      │    │ gripper  │
  │          │    │              │    │ FailDetector │    │          │
  └──────────┘    └──────────────┘    └──────────────┘    └──────────┘
                                              │
                                              ▼ (실패 시)
                                      ┌──────────────┐
                                      │ Plan C       │
                                      │ Classical    │
                                      │ Fallback     │
                                      └──────────────┘
```

---

## Level 0 — 기본 동작 (첫 목표)

> **목표**: 로봇이 움직여서 물체를 잡고 놓는다. 그게 전부.

### 무엇을 하는가

```
하드코딩된 instruction ──▶ VLA inference ──▶ Action Adapter ──▶ 로봇 동작
```

### 구성 요소

| 모듈 | 파일 | 상태 |
|------|------|------|
| VLA Inference Client | `utils/doosan_vla_controller.py` | ✅ 완료 |
| Action Adapter (3중 safety clamp) | `utils/doosan_action_adapter.py` | ✅ 완료 |
| 카메라 캡처 | `utils/doosan_recorder.py` | ✅ 완료 |
| Robot Interface (ROS2/Mock) | `utils/doosan_vla_controller.py` | ✅ 완료 |

### 실행

```bash
python utils/doosan_vla_controller.py \
  --vla-url http://localhost:5555 \
  --instruction "Pick up the blue bottle and place it on the tray"
```

### 성공 기준

- [ ] 로봇이 물체 방향으로 이동
- [ ] 그리퍼가 물체 위치에서 닫힘
- [ ] 물체를 들어올려 목표 위치로 이동
- [ ] 목표 위치에서 그리퍼 열림
- [ ] 성공률 **3/5회 이상**

### 실패 시 체크포인트

```
동작 안 함 ──▶ VLA 서버 연결 확인, joint state 수신 확인
움직이지만 엉뚱한 방향 ──▶ 카메라 위치/각도 조정, 데모 추가 수집
물체 근처까지 가지만 못 잡음 ──▶ gripper threshold 조정, 데모에서 grasp 구간 확인
잡았다가 떨어뜨림 ──▶ gripper 제어 타이밍 확인
```

---

## Level 1 — Action 안정화

> **목표**: 떨림 없는 부드러운 동작. 성공률을 높인다.

### Level 0 → 1 변경점

```diff
  Level 0: VLA chunk → 첫 action만 사용
+ Level 1: VLA chunk → TemporalBlender로 부드럽게 실행
+          파라미터 튜닝으로 동작 품질 개선
```

### TemporalBlender 동작 원리

```
시간 ──────────────────────────────────────────────▶

Chunk 1:  [a0  a1  a2  a3 | a4  a5  a6  a7  ...]
          ───────────────   ─────────────────────
          실행 (horizon=4)   버퍼에 저장 (잔여)

Chunk 2:           [b0  b1  b2  b3 | b4  b5  ...]
                    │   │   │   │
                    ▼   ▼   ▼   ▼
Blending:    w_old·a4 + w_new·b0  (decay^1 = 0.7)
             w_old·a5 + w_new·b1  (decay^2 = 0.49)
             w_old·a6 + w_new·b2  (decay^3 = 0.34)
             w_old·a7 + w_new·b3  (decay^4 = 0.24)
                    │
                    ▼
          부드러운 전환, 급격한 변화 방지
```

### 튜닝 가이드

| 파라미터 | 기본값 | 효과 | 조절 방향 |
|---------|--------|------|----------|
| `execute_horizon` | 4 | 한 번에 실행할 action 수 | 떨리면 ↑, 반응 느리면 ↓ |
| `overlap` | 4 | blending 구간 길이 | 전환이 거칠면 ↑ |
| `decay` | 0.7 | 이전 action 신뢰도 | 관성 크면 ↓, 불안정하면 ↑ |
| `max_delta_per_step` | 0.05 rad | step당 최대 변화 | 느리면 ↑, 위험하면 ↓ |
| `max_joint_velocity` | 1.0 rad/s | 최대 속도 | 느리면 ↑ (주의!) |

### 실행

```bash
# 보수적 (안전, 느림)
python utils/doosan_vla_controller.py --execute-horizon 6 --overlap 4 --decay 0.8

# 공격적 (빠름, 덜 부드러움)
python utils/doosan_vla_controller.py --execute-horizon 3 --overlap 2 --decay 0.5
```

### 성공 기준

- [ ] Level 0 대비 동작이 눈에 띄게 부드러움
- [ ] clamp_ratio **30% 이하** (모델 출력이 안전 범위 내)
- [ ] 성공률 **4/5회 이상**

---

## Level 2 — 음성 명령 (STT)

> **목표**: 하드코딩 대신 말로 지시한다. 시연 임팩트 상승.

### Level 1 → 2 변경점

```diff
  Level 1: 터미널에서 instruction 입력
+ Level 2: 마이크로 말하면 Whisper가 인식
+          (선택) LLM이 한국어 → 영어 robot instruction으로 변환
```

### 파이프라인

```
🎙️ 마이크       Whisper          (선택) Claude       VLA
   │       ──────▶ "파란 병 옮겨" ──────▶ "Pick up   ──────▶ action
   │        base model              the blue bottle"
   5초 녹음     한국어 인식          haiku (빠름)
```

### 구성 요소

| 모듈 | 파일 | 상태 |
|------|------|------|
| STT (Whisper + LLM 정제) | `utils/stt_instruction.py` | ✅ 완료 |
| Controller 통합 | `utils/doosan_vla_controller.py` | ✅ 완료 |

### 실행

```bash
# Whisper만 (영어로 말하기)
python utils/doosan_vla_controller.py --stt

# Whisper + LLM (한국어로 말해도 됨)
python utils/doosan_vla_controller.py --stt --stt-llm

# 녹음 시간 조절
python utils/doosan_vla_controller.py --stt --stt-duration 8
```

### 의존성

```bash
pip install openai-whisper sounddevice
# LLM 정제 사용 시
export ANTHROPIC_API_KEY=sk-...
```

### 성공 기준

- [ ] 한국어 음성이 정확한 영어 instruction으로 변환됨
- [ ] STT 지연 **2초 이내** (base 모델 기준)
- [ ] Level 1과 동일한 성공률 유지
- [ ] 시연 시 관객이 "오!" 하는 반응

### 주의사항

```
⚠️  소음 환경 (대회장)에서 Whisper 정확도 저하 가능
    → fallback: sounddevice 없으면 자동으로 텍스트 입력 모드 전환
    → 조용한 곳에서 녹음하거나, 마이크를 입에 가까이

⚠️  Whisper 모델 크기별 속도/정확도 트레이드오프
    tiny:   빠름 (0.5초), 정확도 낮음
    base:   적당 (1~2초), 대부분 OK  ← 기본값
    small:  느림 (3~5초), 정확도 높음
```

---

## Level 3 — 대화형 인터랙션

> **목표**: 로봇이 상태를 말해준다. 자연어로 대화하듯 제어.

### Level 2 → 3 변경점

```diff
  Level 2: 한 번 말하면 끝까지 실행
+ Level 3: 로봇이 TTS로 상태 피드백
+          연속 대화로 task 수정 가능
+          LLM이 복잡한 지시를 분해
```

### 인터랙션 시나리오

```
사용자: "테이블 위에 있는 거 정리해줘"
  │
  ▼
LLM (Claude): "테이블 위에 파란 병과 빨간 컵이 보입니다.
               어디에 놓을까요?"        ← TTS로 음성 출력
  │
  ▼
사용자: "병은 왼쪽 트레이에, 컵은 오른쪽에"
  │
  ▼
LLM: instruction 분해
  ├─ Step 1: "Pick up the blue bottle and place it on the left tray"
  └─ Step 2: "Pick up the red cup and place it on the right tray"
  │
  ▼
순차 실행, 각 step 완료 시 "완료했습니다" TTS 피드백
```

### 구현 TODO

| 항목 | 난이도 | 우선순위 |
|------|--------|---------|
| TTS 출력 (gTTS / edge-tts) | 낮음 | ★★★ |
| 연속 대화 루프 (STT → LLM → VLA → 반복) | 중간 | ★★☆ |
| LLM instruction 분해 | 중간 | ★★☆ |
| Claude Vision으로 물체 인식 → 대화에 반영 | 높음 | ★☆☆ |

### 성공 기준

- [ ] 로봇이 음성으로 상태 피드백
- [ ] 연속 지시 가능 (1회성이 아님)
- [ ] 시연에서 **대화하듯** 로봇을 제어하는 모습

### 미구현 — 시간 여유 있을 때

```
현재 상태: 설계만 완료
구현 예상: ~2시간
의존성: pip install gtts 또는 edge-tts
```

---

## Level 4 — 실패 감지 & 자동 복구

> **목표**: 실패해도 스스로 재시도하거나 fallback으로 전환. 시연 중 멈추지 않는다.

### Level 3 → 4 변경점

```diff
  Level 3: 실패하면 그냥 멈춤
+ Level 4: 실패를 자동 감지
+          재시도 2회 → 그래도 안 되면 classical fallback
+          시연 중 "끊김 없는" 동작
```

### 실패 감지 로직

```
매 스텝 모니터링
    │
    ├── Stall 감지
    │   10스텝 연속 joint 변화 < 0.002 rad?
    │   ├── Yes → "로봇이 멈췄다"
    │   └── No  → OK
    │
    ├── Over-clamp 감지
    │   5스텝 연속 clamp_ratio > 80%?
    │   ├── Yes → "모델이 비현실적 action 출력 중"
    │   └── No  → OK
    │
    └── 판단
        실패 감지됨?
        ├── retry_count < 2
        │   └── 🔄 TemporalBlender 리셋 → 새 inference부터 재시작
        └── retry_count >= 2
            └── ⚠️ Classical Fallback 전환
                Claude Vision → 물체 인식 → Waypoint → servoj
```

### Classical Fallback (Plan C) 동작

```
카메라 이미지
    │
    ▼
Claude Vision API
    "이미지에서 파란 병의 위치를 찾아주세요"
    │
    ▼
{object: "blue bottle", px: 320, py: 240}
    │
    ▼
pixel_to_robot() 좌표 변환
    │
    ▼
Hardcoded waypoint 궤적
    approach → pick → grasp → lift → move → place → release → home
    │
    ▼
servoj 실행
```

### 구성 요소

| 모듈 | 파일 | 상태 |
|------|------|------|
| Failure Detector | `utils/failure_detector.py` | ✅ 완료 |
| Classical Fallback | `utils/plan_c_classical.py` | ✅ 완료 |
| Controller 통합 | `utils/doosan_vla_controller.py` | ✅ 완료 |

### 실행

```bash
# 자동으로 동작 — 별도 플래그 불필요
python utils/doosan_vla_controller.py --stt --stt-llm

# 콘솔 출력 예시:
#   [step 0] clamp=12% dt=95ms
#   [step 20] clamp=15% dt=89ms
#   🔄 [step 35] stall 감지 — 재시도 #1
#   [step 35] clamp=8% dt=92ms
#   ...
# 또는:
#   🔄 [step 50] over-clamp 감지 — 재시도 #2
#   ⚠️  [step 65] 반복 실패 — classical fallback 전환
#   [Vision] blue bottle at (312, 245)
#   approach → pick → grasp → lift → ...
#   ✅ Done
```

### 성공 기준

- [ ] Stall 상황에서 3초 내 재시도 시작
- [ ] 2회 재시도 후 자동 fallback 전환
- [ ] 시연 중 **한 번도 수동 개입 없이** task 완료
- [ ] 전체 성공률 (재시도/fallback 포함) **4/5회 이상**

---

## 진행 체크리스트

```
현재 위치: ████████░░░░░░░░  Level 0~1 코드 완료, 현장 테스트 대기

Level 0  ████████████████  코드 완료        → 현장에서 첫 동작 확인
Level 1  ████████████████  코드 완료        → 파라미터 튜닝은 현장에서
Level 2  ████████████████  코드 완료        → Whisper + sounddevice 설치 필요
Level 3  ████░░░░░░░░░░░░  설계 완료        → TTS + 대화 루프 구현 필요
Level 4  ████████████████  코드 완료        → 현장에서 threshold 조정
```

---

## 시간 배분 가이드

```
대회 시작
  │
  ├── 0~1h: 환경 셋업 + 로봇 연결 + 카메라 고정
  │
  ├── 1~2.5h: 데모 수집 (50~70 에피소드)
  │         ※ 이 데이터로 모든 레벨을 커버
  │
  ├── 2.5~4h: 데이터 변환 + Fine-tuning 시작 (GPU가 돌리는 동안...)
  │         ├── 동시에: Level 2 STT 테스트
  │         └── 동시에: 카메라 위치 미세 조정
  │
  ├── 4~6h: Level 0 첫 동작 확인
  │         ├── 성공 → Level 1 파라미터 튜닝
  │         └── 실패 → 데모 추가 수집 or 카메라 위치 변경
  │
  ├── 6~10h: Level 1 안정화 + Level 2 STT 통합
  │
  ├── 10~14h: (여유 있으면) Level 3 대화형 구현
  │
  ├── 14~20h: Level 4 실패 감지 threshold 조정 + 반복 테스트
  │
  └── 20~26h: 시연 리허설 + 발표 준비
              ※ 마지막 4시간은 새 기능 추가 금지!
```

---

## 파일 구조

```
groot/
├── TASK_ROADMAP.md              ← 이 파일
├── HACKATHON_STRATEGY.md        ← 전체 전략 + 타임라인
├── configs/
│   ├── doosan_e0509_config.py   ← 로봇 스펙, 안전 파라미터
│   └── groot_modality_config.py ← GR00T modality 설정
├── utils/
│   ├── doosan_vla_controller.py ← 메인 제어 루프 (L0~L4 통합)
│   ├── doosan_action_adapter.py ← VLA → 로봇 명령 (3중 safety)
│   ├── doosan_recorder.py       ← 데모 수집 (hand-guiding)
│   ├── stt_instruction.py       ← L2: Whisper STT + LLM 정제
│   ├── failure_detector.py      ← L4: stall/clamp 감지
│   ├── plan_c_classical.py      ← L4: Claude Vision fallback
│   └── convert_to_lerobot.py    ← 데이터 변환
├── tests/
│   └── test_adapter.py          ← 단위 테스트 (로봇 없이 실행)
└── scripts/
    └── (학습/추론 스크립트)
```

---

## Quick Reference — 레벨별 실행 명령

```bash
# === Level 0: 기본 동작 ===
python utils/doosan_vla_controller.py \
  --instruction "Pick up the blue bottle and place it on the tray"

# === Level 1: 파라미터 튜닝 ===
python utils/doosan_vla_controller.py \
  --execute-horizon 6 --overlap 3 --decay 0.5

# === Level 2: 음성 명령 ===
python utils/doosan_vla_controller.py --stt --stt-llm

# === Level 3: 대화형 (TODO) ===
# python utils/doosan_vla_controller.py --interactive

# === Level 4: 자동 (항상 활성) ===
# failure_detector가 자동으로 동작
# classical fallback도 자동 전환

# === 풀 스택 (L1 + L2 + L4) ===
python utils/doosan_vla_controller.py \
  --stt --stt-llm \
  --execute-horizon 4 --overlap 4 --decay 0.7
```
