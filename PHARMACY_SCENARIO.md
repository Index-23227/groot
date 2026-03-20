# 약국 조제 보조 로봇 — 시나리오 상세

> **문제**: 한국 약사 1인당 하루 처방전 200~300건 처리.
> 조제 과정에서 약품 오인이 연간 수만 건 발생.
> 특히 야간/주말 1인 약국에서 피로 누적 시 위험.
>
> **해결**: 약사가 음성으로 지시하면 로봇이 약품을 시각으로 식별하여
> 정확한 조제함 슬롯에 배치. 약사는 상담/검수에 집중.

---

## 시연 시나리오

### 기본 흐름

```
약사: "파란 약병을 조제함 1번에 넣어줘"
         │
         ▼
  ┌──────────────┐
  │ STT (Whisper) │  "파란 약병을 조제함 1번에 넣어줘"
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ LLM (Claude) │  → "Pick up the blue medicine bottle
  │  instruction │     and place it in dispensing slot 1"
  │  정제        │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Global Camera│  약병 위치 + 조제함 위치를 영상에서 인식
  │ (45° 상단)   │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ VLA (GR00T)  │  action chunk (16 steps) 출력
  │ + Temporal   │  → 부드러운 동작으로 blending
  │   Blender    │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ E0509 로봇   │  approach → pick → grasp → lift
  │ + Safety     │  → transfer → place → release → home
  │   Clamp      │
  └──────┬───────┘
         ▼
  로봇: (TTS) "파란 약병을 1번 슬롯에 넣었습니다"
```

### 시연 대본 (3분)

```
[0:00] 약사(발표자): "안녕하세요. 약국 조제 보조 로봇을 소개합니다."
       "한국 약사는 하루 200~300건의 처방전을 처리하며,
        피로 누적 시 약품 오인 위험이 있습니다."

[0:30] 약사: "이 로봇은 약사의 음성 명령을 듣고
        약품을 정확한 조제함에 배치합니다."

[0:45] 약사: (로봇에게) "파란 약병을 조제함 1번에 넣어줘"
       → STT 인식 → 로봇 동작 시작
       → 파란 약병 집기 → 1번 슬롯에 놓기

[1:30] 약사: "빨간 약병을 조제함 2번에 넣어줘"
       → 다른 약병, 다른 슬롯 → language grounding 시연

[2:15] 약사: "노란 약상자를 3번에 넣어줘"
       → 약병이 아닌 약상자 → 물체 다양성 시연

[2:45] 약사: "이처럼 약사는 손을 대지 않고 음성만으로
        조제 과정을 지시할 수 있어, 상담과 검수에 집중할 수 있습니다."

[3:00] 끝
```

---

## 시뮬레이션

### Isaac Sim 시뮬레이션 (3D 물리 + 카메라 렌더링)

GPU가 있는 환경에서 Isaac Sim 기반 물리 시뮬레이션:

```bash
# 환경 테스트 (GUI 모드)
python sim/pharmacy_isaac_env.py

# Headless + Domain Randomization
python sim/pharmacy_isaac_env.py --headless --domain-rand

# 자동 데모 수집 (200개, DR 적용)
python sim/isaac_data_collector.py --num-episodes 200 --headless --domain-rand

# 특정 시나리오만
python sim/isaac_data_collector.py --scenario basic_single --num-episodes 50
```

### Isaac Sim → 학습 → 배포 파이프라인

```
Isaac Sim 데모 수집               학습                    배포
─────────────────    ──────────────────    ──────────────
sim/isaac_data_      utils/convert_to_    sim/sim2real_
collector.py         lerobot.py           deploy.py
     │                    │                    │
     ▼                    ▼                    ▼
./data/sim_raw/  →  ./data/lerobot/  →  실제 E0509 로봇
(200~500 에피소드)   (LeRobot v2 형식)   (Sim2Real 보정 적용)
```

```bash
# 1. Sim 데모 수집
python sim/isaac_data_collector.py --num-episodes 500 --headless --domain-rand

# 2. LeRobot v2 변환
python utils/convert_to_lerobot.py --data-dir ./data/sim_raw

# 3. GR00T 학습
bash scripts/04_train_groot.sh

# 4-a. Sim에서 먼저 평가
python sim/sim2real_deploy.py --mode sim-eval --scenario basic_single

# 4-b. Sim2Real 캘리브레이션 (실제 로봇 연결 시)
python sim/sim2real_deploy.py --mode calibrate

# 4-c. 실제 로봇 배포
python sim/sim2real_deploy.py --mode real \
  --instruction "Pick up the blue medicine bottle and place it in dispensing slot 1"
```

### 소프트웨어 시뮬레이션 (로봇 없이)

전체 파이프라인을 로봇 없이 검증:

```bash
# 기본 시나리오
python utils/pharmacy_sim.py --scenario basic_single

# 전체 시나리오 순회
python utils/pharmacy_sim.py --scenario all

# 실패 주입 테스트
python utils/pharmacy_sim.py --inject-failure stall    # stall → 재시도 → fallback
python utils/pharmacy_sim.py --inject-failure clamp    # over-clamp → 재시도 → fallback

# STT 시뮬레이션 (텍스트 입력)
python utils/pharmacy_sim.py --stt

# 파라미터 튜닝 테스트
python utils/pharmacy_sim.py --execute-horizon 6 --overlap 3 --decay 0.5
```

### 시뮬레이션이 검증하는 것

| 항목 | 검증 내용 |
|------|----------|
| MockVLA → TemporalBlender | chunk blending이 부드러운 동작을 만드는가 |
| ActionAdapter 3중 safety | delta/position/velocity clamp이 정상 동작하는가 |
| FailureDetector | stall/over-clamp을 정확히 감지하는가 |
| 재시도 로직 | blender reset 후 새 inference가 정상 동작하는가 |
| Fallback 전환 | 2회 실패 후 classical fallback으로 넘어가는가 |
| STT 파이프라인 | instruction 입력이 올바르게 전달되는가 |

### 시나리오 목록

| ID | 설명 | 약품 | 슬롯 |
|----|------|------|------|
| `basic_single` | 기본 — 파란 약병 | blue_bottle | 1 |
| `basic_red` | 기본 — 빨간 약병 | red_bottle | 2 |
| `basic_white` | 기본 — 하얀 약병 | white_bottle | 3 |
| `different_slot` | 같은 약 다른 슬롯 | blue_bottle | 3 |
| `box_item` | 약상자 (다른 형태) | yellow_box | 1 |

---

## 레벨별 시연 포인트

### Level 0 — "동작합니다"

```
하드코딩 instruction → 로봇이 약병을 잡고 놓음
심사위원에게: "로봇이 VLA 모델의 출력으로 실제 약품을 조작합니다"
```

### Level 1 — "정밀합니다"

```
TemporalBlender로 부드러운 동작 → 약병을 흔들리지 않게 옮김
심사위원에게: "action chunk blending으로 약품 취급에 적합한 정밀 동작을 구현했습니다"
```

### Level 2 — "말로 지시합니다"

```
약사가 마이크에 한국어로 말함 → Whisper → Claude → VLA
심사위원에게: "약사가 조제대에서 손을 대지 않고 음성만으로 로봇을 제어합니다"
```

### Level 3 — "대화합니다"

```
로봇이 TTS로 "파란 약병을 1번에 넣었습니다" 피드백
심사위원에게: "양방향 인터랙션으로 약사가 조제 상황을 실시간 파악합니다"
```

### Level 4 — "실수해도 복구합니다"

```
의도적으로 약병 위치를 바꿔놓기 → 로봇이 재시도 → 성공
심사위원에게: "약품 오인 시 자동 감지하고 재시도하여 안전성을 보장합니다"
```

---

## 사회적 맥락 — 발표 자료용

### 왜 의미 있는가

```
1. 약사의 단순 반복 노동 경감
   → 하루 200~300건 처방전 중 조제 배치는 반복 작업
   → 로봇에 위임하면 약사는 상담/검수에 집중

2. 약품 오인 방지
   → VLM의 language grounding: "파란 약병"을 정확히 시각 식별
   → 사람의 피로에 의한 오인 vs 모델의 일관된 인식

3. 고령 사회 대응
   → 약국 수 감소 + 1인 약국 증가
   → 야간/주말 1인 운영 시 보조 로봇 수요 급증

4. 확장성
   → 약국 뿐 아니라 병원 조제실, 물류 분류 등에 적용 가능
```

### 수치 (발표 슬라이드용)

```
- 한국 약국 수: 약 24,000개 (2024)
- 약사 1인당 하루 처방전: 200~300건
- 약품 오인 사고: 연간 수만 건 (KOPS 보고)
- 1인 약국 비율: 증가 추세
- 야간 약국: 전체의 ~10%
```
