# GR00T — SAM2 + VLM CoT Pick & Place Pipeline

> **해커톤**: 커널아카데미 Physical AI 26H Robotics Hackathon (2026.03.20~21)
> **로봇**: Doosan E0509 6-DOF + ROBOTIS RH-P12-RN-A Gripper
> **카메라**: Intel RealSense D435 (1280x720, 30fps)
> **GPU**: 3090 x 8
> **태스크**: 자연어 음성 명령 → Pick & Place (예: "파란색 캔을 초록색 캔 옆에 놓아라")
> **시나리오**: 약국 조제 보조 (투약 오류 방지)

---

## 아키텍처 개요

```
음성 입력 (Whisper STT)
        ↓
자연어 명령 ("파란색 캔을 초록색 캔 옆에 놓아라")
        ↓
┌─────────────────────────────────────────────┐
│  Perception: SAM2-tiny Segmentation         │
│  → ROI 영역 blind segmentation              │
│  → 객체별 mask + crop (배경 흰색 처리)       │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Reasoning: GPT-4o Chain-of-Thought         │
│  Step 1. Task Decomposition (pick/place)    │
│  Step 2. PICK 대상 객체 식별 (crops 비교)    │
│  Step 3. PLACE 참조 객체 식별               │
│  Step 4. 관계 추론 (위에/옆에/안에)          │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Localization: Depth 기반 EE 좌표 계산       │
│  → 윗면 중심 = midpoint(depth최소, y최소)    │
│  → 카메라 3D 좌표 (RealSense deprojection)   │
│  → 캘리브레이션 변환 (camera → robot)        │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Execution: Doosan E0509 ROS2               │
│  → move_line (직선 이동)                     │
│  → gripper_control (잡기/놓기)               │
│  → 안전 높이 경유 + home 복귀               │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Verification: Before/After 비교 (GPT-4o)   │
│  → 성공/실패 판단 + 실패 원인 분류           │
│  → Error Recovery (EE 보정, 재시도)          │
└─────────────────────────────────────────────┘
```

---

## 실행 방법

### 1. LLM 파이프라인 테스트 (로봇 없이)

```bash
# SAM2 + GPT-4o CoT 파이프라인 시각화 + 검증
python tests/test_vis_sam2_cot.py
```

- 저장된 이미지(`data/base_images/`)로 전체 파이프라인 실행
- Task Decomposition → 객체 식별 → EE 좌표 계산 → HTML 리포트 생성
- Action Verification (before/after 비교)으로 성공 여부 판단
- 세션 기반 상태 관리 (TaskManager)로 multi-step 태스크 추적
- 결과: `results/vis_sam2_cot/` 에 HTML 시각화 저장

### 2. 로봇 실 테스트 (Doosan E0509 연동)

```bash
# SAM2 + GPT-4o + 음성인식 + 로봇 제어 (전체 파이프라인)
python WJ/groot_pick_and_place.py
```

- Enter → 음성 녹음 → Whisper STT → 자연어 명령 획득
- RealSense 카메라 실시간 캡처 → SAM2 세그먼테이션
- GPT-4o CoT 추론 → 캘리브레이션 → ROS2 로봇 제어
- 물체 타입 자동 분류 (캔/블럭) → 그리퍼 힘 자동 조절

---

## 레포 구조

```
groot/
├── README.md
├── requirements.txt
│
├── WJ/                                ⭐ 로봇 실행 스크립트
│   ├── groot_pick_and_place.py        ⭐ 메인: SAM2 + GPT-4o + STT + 로봇 제어
│   ├── pick_and_place_auto.py         높이 기반 자동 분류 pick & place
│   ├── pick_and_place_ai.py           AI 보조 포인트 선택
│   ├── pick_and_place_auto_simple.py  간소화 자동 모드
│   ├── pick_and_place_oneclick.py     원클릭 실행
│   ├── pick_and_place_simple.py       기본 클릭 기반 제어
│   ├── capture_scene.py               장면 캡처 (디버깅용)
│   ├── detect_object.py               객체 감지 테스트
│   └── calibrate_and_control_old.py   레거시 캘리브레이션
│
├── tests/                             ⭐ 파이프라인 테스트 (로봇 없이 실행)
│   ├── test_vis_sam2_cot.py           ⭐ 메인: SAM2 + CoT 전체 파이프라인 시각화
│   ├── test_vis_base12.py             베이스 모델 v12 테스트
│   ├── test_vis_base14.py             베이스 모델 v14 테스트
│   ├── test_gemini_embodied.py        Gemini Embodied Reasoning 검증
│   ├── test_gemini_auto_eval.py       Gemini 자동 평가
│   ├── test_graspability.py           잡기 가능성 평가
│   ├── test_graspability_icl.py       ICL 기반 잡기 평가
│   ├── test_graspability_local.py     로컬 잡기 평가
│   ├── test_pipeline_vis.py           Pipeline A 시각화
│   ├── test_pipeline_html.py          Pipeline HTML 리포트
│   ├── test_compare_ab.py             Pipeline A vs B 비교
│   ├── test_rgbd_pipeline_html.py     RGBD 파이프라인 리포트
│   ├── test_object_detection.py       객체 감지
│   ├── test_object_localizer.py       ObjectLocalizer 테스트
│   ├── test_segmentation.py           SAM2 세그먼테이션
│   ├── test_depth_cluster.py          Depth 클러스터링
│   ├── test_depth_pixel.py            Depth 픽셀 검증
│   └── test_table_roi.py             테이블 ROI 추출
│
├── utils/                             핵심 유틸리티
│   ├── task_manager.py                세션/태스크 상태 관리 + history 추적
│   ├── task_planner.py                Gemini ER 기반 SayCan 대안 (CoT + 동적 재계획)
│   ├── gemini_bridge.py               Gemini → Doosan 실행 브릿지 (closed-loop)
│   ├── gemini_saycan.py               ReAct 기반 오케스트레이터
│   ├── gemini_visualizer.py           4패널 실시간 추론 대시보드
│   ├── object_localizer.py            SAM2 + GPT-4o 다단계 객체 탐지
│   ├── pipeline_a.py                  Depth 클러스터링 → VLM crop 선택
│   ├── pipeline_b.py                  VLM bbox 추출 → Depth 클러스터링
│   ├── rgbd_localizer.py              경량 RGBD 파이프라인 (SAM2 없이)
│   ├── calibration.py                 ArUco 기반 Hand-Eye 캘리브레이션
│   ├── skill_library.py               로봇 스킬 프리미티브 (pick/place/inspect/home)
│   ├── stt_instruction.py             Whisper 기반 음성 입력
│   └── doosan_recorder.py             Direct Teaching 데모 녹화
│
├── configs/
│   ├── doosan_e0509_config.py         ⭐ 중앙 설정 (로봇/카메라/그리퍼/안전)
│   └── __init__.py
│
├── vla/                               VLA 모델 관련
│   ├── doosan_vla_controller.py       VLA 추론 서버 ↔ 로봇 (TemporalBlender)
│   ├── doosan_action_adapter.py       VLA → 로봇 변환 (3중 안전 클램핑)
│   ├── convert_to_lerobot.py          Raw → LeRobot v2 데이터 변환
│   ├── failure_detector.py            런타임 실패 감지 + 자동 재시도
│   ├── groot_modality_config.py       GR00T 6+1 DOF 등록
│   ├── plan_c_classical.py            Plan C: cuRobo fallback
│   ├── pharmacy_scenario.py           약국 시나리오 정의
│   ├── pharmacy_sim.py                소프트웨어 시뮬레이션 (로봇 불필요)
│   ├── test_adapter.py                ActionAdapter 단위 테스트
│   └── sim/
│       ├── pharmacy_isaac_env.py      Isaac Sim 약국 환경
│       ├── isaac_data_collector.py    자동 데모 데이터 수집
│       └── sim2real_deploy.py         Sim-to-Real 전이
│
├── tools/                             시각화 도구
│   ├── draw_bboxes.py                 Bounding box 그리기
│   ├── draw_object_bboxes.py          객체별 bbox 시각화
│   └── visualize_pipeline.py          파이프라인 실행 시각화
│
├── scripts/                           셋업/학습/배포 자동화
│   ├── 01_setup_groot.sh
│   ├── 01_setup_smolvla.sh
│   ├── 02_record_demos.sh
│   ├── 03_convert_data.sh
│   ├── 04_train_groot.sh
│   ├── 04_train_smolvla.sh
│   └── 05_deploy.sh
│
├── data/base_images/                  테스트용 기본 이미지
├── results/                           실행 결과 + 시각화 출력
│   ├── vis_sam2_cot/                  SAM2+CoT HTML 리포트
│   ├── sessions/                      세션 히스토리 (JSON)
│   ├── auto_eval/                     자동 평가 리포트
│   └── ...
│
└── docs/                              문서
    ├── HACKATHON_STRATEGY.md          전략 + 실험 결과
    ├── TASK_ROADMAP.md                Level 0~4 기능 로드맵
    ├── PHARMACY_SCENARIO.md           약국 시나리오 설계
    ├── PHYSICAL_SETUP_GUIDE.md        하드웨어 셋업 가이드
    └── PROJECT_JOURNEY.md             개발 타임라인 로그
```

---

## 핵심 파이프라인 상세

### SAM2 + GPT-4o CoT (test_vis_sam2_cot.py)

전체 LLM 테스트 파이프라인으로, 로봇 없이 이미지 기반으로 동작:

```
Step 0: 원본 이미지 + ROI 설정 (가운데 1/3 영역)
Step 1: SAM2-tiny blind segmentation → 객체별 mask + crop
Step 2a: Task Decomposition (자연어 → pick/place 분해)
Step 2b: PICK 객체 식별 (crops → GPT-4o)
Step 2c: PLACE 참조 객체 식별
Step 3: EE 좌표 계산 — depth 기반 윗면 중심
         Point A: depth 최소점 (카메라에 가장 가까운 면)
         Point B: y 최소점 (이미지 최상단)
         EE = midpoint(A, B)
Step 4: Action Verification (before/after GPT-4o 비교)
         → 실패 시 원인 분류 (ee_offset, grip_fail, collision 등)
         → Error Recovery: EE 위치 보정 후 재시도
```

- **세션 관리**: TaskManager로 multi-step 태스크 추적 (중단 후 이어서 실행 가능)
- **출력**: HTML 시각화 리포트 (ROI, SAM2 overlay, crop grid, EE crosshair)

### 로봇 실행 (groot_pick_and_place.py)

실제 Doosan E0509 로봇과 연동하는 전체 시스템:

```
1. 음성 녹음 (pyaudio) → Whisper large-v3 STT (한국어)
2. RealSense D435 RGB-D 캡처
3. SAM2-tiny 세그먼테이션 (ROI: 좌/우 1/4 제외)
4. GPT-4o CoT 추론 (PICK/PLACE 식별)
5. depth 기반 윗면 중심 → RealSense deprojection → 3D 카메라 좌표
6. 캘리브레이션 변환 (camera → robot, 24포인트 least squares)
7. ROS2 move_line → 안전 높이 경유 → pick → grip → place → release → home
```

- **물체 분류**: z 높이로 캔/블럭 자동 판별 → 그리퍼 힘 자동 조절
- **안전**: MIN_Z=120mm, MAX_REACH=800mm, SAFE_Z=400mm 경유

---

## 전략 계층 (원래 계획 vs 실제 결과)

```
원래 계획:
  A안: GR00T N1.6 (3B VLA)        → 메인
  B안: SmolVLA (450M VLA)          → 백업
  C안: Claude Vision + cuRobo       → 최후의 보루

실제 결과:
  ⭐ SAM2 + GPT-4o CoT + ROS2     → 메인 (Plan C 변형이 가장 잘 동작)
     - 학습 불필요, zero-shot
     - 정확한 depth 기반 EE 계산
     - 음성 입력 + 실시간 추론 + 로봇 제어 통합
```

---

## 설치

```bash
pip install -r requirements.txt

# SAM2 체크포인트 (sam2.1_hiera_tiny.pt)
# → checkpoints/ 디렉토리에 배치

# OpenAI API 키
# → token 파일에 저장 또는 환경변수 OPENAI_API_KEY 설정
```

### 주요 의존성

- `torch`, `torchvision` — SAM2 실행
- `openai` — GPT-4o CoT 추론
- `google-genai` — Gemini Embodied Reasoning (대안 파이프라인)
- `anthropic` — Claude Vision (Plan C)
- `opencv-python`, `Pillow` — 이미지 처리
- `pyrealsense2` — RealSense D435 카메라
- `faster-whisper` — 음성 인식 (STT)
- `pyaudio` — 마이크 녹음

---

## 팀 역할

| | 역할 | 핵심 담당 |
|--|------|----------|
| **P1** | 리드 | 로봇 셋업 + ROS2 + 캘리브레이션 + 전략 의사결정 + 발표 |
| **P2** | A안 | GR00T 셋업 + 학습/튜닝 |
| **P3** | B안 | SmolVLA + 데이터 변환 |
| **P4** | 인프라 | 그리퍼 + 카메라 + cuRobo + Isaac Sim |
| **P5** | C안 | VLM 파이프라인 구현 + 시연 시나리오 |
