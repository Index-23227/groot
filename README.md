# 🤖 Physical AI Hackathon — Doosan E0509 VLA Pipeline

> **해커톤**: 커널아카데미 Physical AI 26H Robotics Hackathon (2026.03.20~21)
> **로봇**: Doosan E-Series 6-DOF + Gripper
> **GPU**: 3090 × 8
> **태스크**: 자연어 명령 → Pick & Place (예: "파란 약병을 트레이에 옮겨줘")
> **시나리오**: 약국 조제 보조 (투약 오류 방지)

---

## 전략 계층

```
A안:     GR00T N1.6 (3B)     GPU 4장   → 메인 VLA
Plan B:  SmolVLA (450M)      GPU 1장   → 백업 VLA (먼저 결과 나옴 → baseline)
Plan C:  Claude Vision+cuRobo GPU 0장   → VLA 전부 실패 시 최후의 보루
```

전부 AI 사용: GR00T(VLA) / SmolVLA(VLA) / Claude Vision(LLM)

---

## 📁 레포 구조

```
hackathon-doosan-vla/
├── README.md                          ← 이 파일
├── requirements.txt
├── configs/
│   ├── doosan_e0509_config.py         ⭐ 중앙 설정 (현장에서 이것만 수정)
│   └── groot_modality_config.py       GR00T embodiment 등록용
├── scripts/
│   ├── 01_setup_groot.sh              P2: GR00T 환경 셋업
│   ├── 01_setup_smolvla.sh            P3: SmolVLA 환경 셋업
│   ├── 02_record_demos.sh             P1+P2: 데모 녹화
│   ├── 03_convert_data.sh             P3: 데이터 변환
│   ├── 04_train_groot.sh              P2: GR00T 학습
│   ├── 04_train_smolvla.sh            P3: SmolVLA 학습
│   └── 05_deploy.sh                   배포 (groot / smolvla / planc)
├── utils/
│   ├── doosan_recorder.py             ⭐ 데모 녹화 (Direct Teaching + 카메라)
│   ├── convert_to_lerobot.py          ⭐ Raw → GR00T LeRobot v2 변환
│   ├── doosan_action_adapter.py       ⭐ VLA → 로봇 (safety clamp)
│   ├── doosan_vla_controller.py       배포: inference server ↔ 로봇
│   └── plan_c_classical.py            Plan C: Claude Vision + cuRobo
└── tests/
    └── test_adapter.py                단위 테스트 (로봇 없이 실행)
```

---

## 👥 5인 팀 역할

| | 역할 | 핵심 담당 |
|--|------|----------|
| **P1** | 리드 | 로봇 셋업 + ROS2 조작 + 데모(조작자) + 전략 의사결정 + 발표 |
| **P2** | A안 | GR00T 셋업 + 데모(운영자) + GR00T 학습/튜닝 |
| **P3** | Plan B | SmolVLA 셋업 + 데이터 변환 + SmolVLA 학습 |
| **P4** | 인프라 | 그리퍼 + 환경설정 + 카메라 뷰 + quaternion + cuRobo + Isaac Sim 세팅 |
| **P5** | Plan C | Claude+cuRobo 구현 + 시연 시나리오 + 발표 자료 |

---

## 🚀 Phase 0: 현장 셋업 [시간 0~2h]

### P1 — 로봇 확인 + config 업데이트

```bash
# 체크리스트
# [ ] 두산 모델 확인 (E0509? 다른 모델?)
# [ ] 컨트롤러 IP (기본: 192.168.127.100)
# [ ] teach pendant → joint limits 확인
# [ ] hand-guiding 버튼 동작 확인
# [ ] 그리퍼 종류 + 제어 방식 (joint? GPIO? Modbus?)
# [ ] configs/doosan_e0509_config.py 업데이트
```

### P2 — GR00T 환경 셋업 (~2h)

```bash
bash scripts/01_setup_groot.sh
# 내부:
# 1. Isaac-GR00T 클론 + pip install (~30분)
# 2. flash-attn 설치 (~15분, 컴파일)
# 3. GR00T N1.6 checkpoint 다운로드 (~20분)
# 4. 두산 modality config 배치
# 5. ⭐ 더미 데이터로 1 step 학습 테스트 → 성공 확인 필수
```

### P3 — SmolVLA 환경 셋업 (~30분)

```bash
bash scripts/01_setup_smolvla.sh
# 내부:
# 1. LeRobot 클론 + pip install -e ".[smolvla]" (~10분)
# 2. SmolVLA base 모델 로드 테스트 (~10분)
# 3. 끝. 남는 시간에 데이터 변환 dry-run 준비
```

### P4 — 카메라 + ROS2 + 녹화 환경

```bash
# 카메라 연결 확인
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"

# ROS2 두산 노드 실행
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real model:=e0509 host:=192.168.127.100

# joint_states 확인
ros2 topic echo /dsr01e0509/joint_states

# doosan_recorder.py의 카메라/ROS2 topic 현장 맞춤 수정
# 테스트 녹화 2~3회 → 데이터 shape 확인
```

### P5 — Plan C (Claude + cuRobo) 구현 시작

```bash
# Claude Vision API 테스트
python utils/plan_c_classical.py --test-vision

# cuRobo 설치 + URDF 로드 + IK 테스트
```

---

## 🎬 Phase 1: 데모 수집 [시간 2~4h]

### P1(조작자) + P2(운영자) 2인 체제

```bash
# P2가 스크립트 실행:
bash scripts/02_record_demos.sh "Pick up the blue bottle and place it on the tray" 50

# P1: teach pendant의 hand-guiding 버튼을 누른 채로
#     로봇 팔을 잡고 pick & place 동작 수행
# P2: Enter로 녹화 시작, Ctrl+C로 종료, 물체 위치 리셋
```

물체 위치 5곳 × 10회 = 50 episodes (~2h):
```
    ┌─────────────────────────┐
    │         ⑤ (뒤)          │
    │   ①        ②       ③   │
    │  (왼)    (중앙)    (오)  │
    │         ④ (앞)          │
    │      🤖 로봇 베이스      │
    └─────────────────────────┘
```

### P3 — 초기 에피소드로 변환 파이프라인 테스트
### P4 — 녹화 모니터링 + 데이터 QA
### P5 — Plan C 구현 계속

---

## 🔄 Phase 2: 데이터 변환 [시간 4~5h]

### P3 (주담당) + P4 (지원)

```bash
bash scripts/03_convert_data.sh "Pick up the blue bottle and place it on the tray"
# 내부:
# 1. 실패 에피소드 제거
# 2. Raw npz → GR00T LeRobot v2 변환
# 3. modality.json 생성
# 4. 검증 (info.json, stats.json 확인)
# → P2에게 "데이터 준비 완료" 알림
```

---

## 🧠 Phase 3: 학습 [시간 5~12h]

### P2 — GR00T 학습 (GPU 0~3)

```bash
bash scripts/04_train_groot.sh
# 10000 steps, batch 64, 4 GPUs
# WandB 모니터링, 2K/5K/10K checkpoint 저장
# 8h쯤 5K checkpoint open-loop 평가
```

### P3 — SmolVLA 학습 (GPU 4) ← 동시 진행!

```bash
bash scripts/04_train_smolvla.sh
# 20000 steps, batch 64, 1 GPU
# GR00T보다 먼저 결과 나옴 → baseline 역할
```

### P4 — cuRobo + Action Adapter 준비

```bash
# cuRobo URDF + IK + collision mesh
# Action adapter ↔ 실물 로봇 통신 테스트
# dummy action → servoj → 로봇 1° 움직이는지 확인
```

### P5 — Plan C 완성 + 시연 설계 시작

---

## ⚡ Phase 3.5: 핵심 결정 [시간 10~12h]

```
P1이 GR00T vs SmolVLA open-loop 결과 비교 → 메인 모델 결정

시나리오 A: GR00T 승     → GR00T 배포, SmolVLA는 비교 대상
시나리오 B: SmolVLA 승   → SmolVLA 배포, GR00T 재학습
시나리오 C: 둘 다 OK     → 둘 다 배포 → "VLA 비교" 발표 (가장 강력)
시나리오 D: 둘 다 실패   → Plan C 즉시 투입
```

---

## 🔧 Phase 4: 배포 + 디버깅 [시간 12~22h]

```bash
# GR00T 배포
bash scripts/05_deploy.sh groot

# SmolVLA 배포
bash scripts/05_deploy.sh smolvla

# Plan C 배포
bash scripts/05_deploy.sh planc
```

**예상 문제 + 해결:**
- 로봇 jittery → `MAX_DELTA_PER_STEP` 줄이기 (0.05→0.03)
- Action drift → `ACTION_HORIZON` 줄이기 (16→8→4)
- 통신 지연 → 제어 주파수 낮추기 (10Hz→5Hz)
- 학습 발산 → Ctrl+C 후 재시작 (GR00T 알려진 이슈)
- 그리퍼 안 움직임 → `GRIPPER_TYPE` 확인, GPIO/Modbus 별도 구현

---

## 🎤 Phase 5: 발표 준비 [시간 22~26h]

P1(기술 발표) + P5(자료 제작) 역할 분담

발표 구성:
1. 문제 정의 — healthcare pick & place, 투약 오류 방지
2. 기술 아키텍처 — VLA + cuRobo hybrid pipeline
3. 라이브 데모 — 자연어 명령 → 로봇 동작
4. 결과 비교 — GR00T vs SmolVLA (또는 VLA vs Classical)
5. 향후 계획
