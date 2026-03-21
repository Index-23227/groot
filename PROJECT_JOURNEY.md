# GROOT 프로젝트 여정 기록

> Doosan E0509 협동로봇 + VLA/VLM 기반 약국 자동 조제 시스템
> 커널아카데미 Physical AI 해커톤 (2026.03.19 ~ 03.21, 26시간)

---

## 팀 구성

| 이름 | 역할 |
|------|------|
| Hyeongjin Kim | VLM 엔지니어링, 코드베이스 관리 |
| jun | Gemini ER/SAM2/GPT-4o 파이프라인 개발 |
| JJW2602 | 로컬 VLM 테스트 (Cosmos-Reason2), STT 통합 |
| woongje-cho | 로봇 제어, 캘리브레이션, 현장 통합 |
| Claude | 코드 구조화, 문서화, 하드웨어 설정 수정 |

---

## Phase 1 — 프로젝트 초기 셋업 (3/19)

### 해커톤 전날 준비

- **18:10** — Hyeongjin Kim이 초기 파일 업로드 (hackathon_pre_prep.md, hackathon_groot_v100_guide.md 등)
- **19:24** — 추가 파일 업로드
- **14:51** — 해커톤 전략 문서 작성 (`HACKATHON_STRATEGY.md`)
  - 26시간 타임라인, 아키텍처 설계, 리스크 대응 전략
  - Plan A (GR00T N1.6 VLA) / Plan B (SmolVLA) / Plan C (Claude Vision + cuRobo) 3단계 폴백 전략 수립

### 핵심 결정사항
- **로봇**: Doosan E0509 (6-DOF, 900mm reach, 5kg payload)
- **그리퍼**: ROBOTIS RH-P12-RN-A
- **카메라**: Intel RealSense (1280×720 @ 30fps)
- **시나리오**: 약국 조제 보조 — "파란 약병을 조제함 1번에 넣어줘"

---

## Phase 2 — 코드베이스 구조화 및 문서화 (3/20)

### 오전: 프로젝트 모듈화

- **09:12** — tar 파일 추출 및 프로젝트 모듈 구조로 재조직
  - `configs/`, `utils/`, `scripts/`, `tests/` 디렉토리 체계 수립
- **09:14** — P1 리드 역할에 ROS2 운영 책임 추가
- **09:17** — P4 역할 업데이트 (그리퍼, 환경설정, 카메라뷰, 쿼터니언, cuRobo, Isaac Sim)
- **→ PR #1 머지**

### 오전~오후: 핵심 모듈 개발

- **09:33** — `TemporalBlender` 테스트 추가 (first_chunk, smooth, reset)
- **09:34** — `TemporalBlender` 구현 — VLA action chunk의 시간적 블렌딩으로 부드러운 로봇 모션 구현
  - Execution horizon: 4 steps, Overlap: 4 steps, Decay: 0.7
- **→ PR #2 머지**

### 오후: 기능 확장 및 로드맵

- **12:40** — STT 음성 입력 (Level 2) 및 실패 감지 (Level 4) 추가
  - `stt_instruction.py`: Whisper 기반 음성→텍스트
  - `failure_detector.py`: Stall/Over-clamp 감지
- **12:45** — `TASK_ROADMAP.md` 작성 — Level 0~4 단계별 완성도 로드맵
- **→ PR #3 머지**

### 오후: 시뮬레이션 및 물리 환경

- **12:55** — 약국 시나리오 시뮬레이션 + 물리 셋업 가이드 추가
  - `pharmacy_sim.py`: 소프트웨어 시뮬레이션 (MockVLA, 3-tier safety clamp 검증)
  - `PHYSICAL_SETUP_GUIDE.md`: 하드웨어 배치 가이드
- **→ PR #4 머지**

- **13:24** — Isaac Sim 환경, 데이터 수집기, Sim-to-Real 배포 추가
  - `sim/pharmacy_isaac_env.py`: 물리 기반 시뮬레이션
  - `sim/isaac_data_collector.py`: 도메인 랜덤화 포함 자동 데이터 수집
  - `sim/sim2real_deploy.py`: Sim→Real 브릿지
- **13:35** — 물리 셋업 가이드를 실제 현장 배치에 맞게 업데이트
- **→ PR #5 머지**

### 오후: 문서 보강 및 하드웨어 수정

- **13:42** — `PHARMACY_SCENARIO.md` 강화 — VLA를 쓰는 이유, E0509를 쓰는 이유 서사 보강
  - 기존 자동화 시스템 $1-3M vs 협동로봇의 접근성
  - 한국 2.4만 약국 대상 사회적 임팩트
- **→ PR #6 머지**

- **13:53~14:01** — 하드웨어 불일치 3연속 수정
  - `doosan_e0509_config.py`: 그리퍼 스펙, 조인트 리밋 정밀 보정
  - `doosan_action_adapter.py`: 3-tier 안전 클램프 강화
  - `hackathon_pre_prep.md`: 그리퍼 컨벤션 일관성 수정
- **→ PR #7 머지**

- **23:11** — `track2-gemini-curobo.tar.gz` 업로드 (Track 2용 Gemini+cuRobo 코드)

---

## Phase 3 — Gemini ER 파이프라인 개발 (3/21 새벽 02:00~04:30)

### Gemini Embodied Reasoning 테스트

- **02:25** — Gemini ER 1.5 체화 추론 테스트 & 자동 평가 파이프라인
  - `test_gemini_embodied.py` (1,174줄): 종합 테스트
  - `test_gemini_auto_eval.py` (803줄): 자동 평가
  - `gemini_visualizer.py`: 시각화 도구
- **02:32** — 평가 결과 및 HTML 브리핑 리포트 추가
  - affordance_pass, manipulation_fail, manipulation_partial 결과 생성

### 코드 재구성 및 핵심 유틸리티

- **02:49** — VLA 모듈을 `vla/` 디렉토리로 재조직
  - `gemini_bridge.py` (747줄): Gemini ER → 로봇 실행 브릿지 (5단계 closed-loop)
  - `calibration.py` (333줄): 카메라-로봇 좌표 변환 (Hand-Eye Calibration)
- **03:09** — `GeminiSayCan` 추가: ReAct(Reason-Act-Observe) 패턴 기반 태스크 오케스트레이터
  - `gemini_saycan.py`: PLAN → AFFORD → ACT → OBSERVE → VERIFY 루프
  - `skill_library.py`: 스크립트 기반 조작 스킬 라이브러리
  - `task_planner.py`: 태스크 분해 로직

### Graspability(파지 가능성) 평가

- **03:30** — Gemini ER 1.5 grasp feasibility 평가 시작
- **03:47** — 테스트용 base 이미지 9장 (base1~9) 추가
- **03:54** — Graspability 테스트 리포트 (MD) 생성
- **03:56** — Ground Truth 기반 정밀 평가 추가
- **04:02** — (JJW2602) 로컬 VLM 테스트: Cosmos-Reason2 2B/8B 모델 비교 평가
- **04:31** — ICL(In-Context Learning) 적용: **Precision 50% → 75%** 개선 (4-shot examples)

---

## Phase 4 — SAM2 + VLM 파이프라인 고도화 (3/21 오전 04:30~13:00)

### 객체 탐지 파이프라인 진화

- **04:36** — Bbox 시각화: Gemini ER 탐지 결과를 base 이미지에 오버레이
- **04:55** — Object-only 탐지: 그리퍼 제외로 hallucination 감소
- **05:26** — Instruction 기반 단일 타겟 탐지로 전환 (sprite can 사례)

### SAM2 + Gemini → GPT-4o 전환

- **06:04** — SAM2 → Gemini ER 객체 로컬라이제이션 파이프라인 구축
  - `object_localizer.py`: SAM2-tiny blind segmentation + VLM reasoning
- **06:35** — SAM2 + Gemini 파이프라인 시각화 및 depth pixel 테스트
- **06:40** — **파이프라인 최적화: 20분 → 53초** 🔥
  - SAM2 캐시, top-5 필터, parallel Gemini 호출
- **06:46** — base10~12 이미지 추가 (노란 실린더 pick & place 장면)
- **08:11** — **Gemini ER → GPT-4o 교체** + SAM2-free RGBD 파이프라인 추가
  - `rgbd_localizer.py` (346줄): 깊이 정보 기반 객체 로컬라이제이션

### 파이프라인 비교 및 CoT 추론

- **09:04** — Pipeline A/B 비교 및 depth-only 클러스터링 실험
  - `pipeline_a.py`, `pipeline_b.py`: 두 접근법 비교 프레임워크
- **11:14** — SAM2 + CoT 파이프라인 (ROI 기반 세그멘테이션)
  - base14, base15 depth 이미지 추가
  - `test_vis_sam2_cot.py` (534줄): 핵심 시각화 테스트
- **12:13** — Pick-and-Place CoT with geometric End-Effector 계산
- **13:19** — Scene-object-free CoT로 파이프라인 업데이트 (일반화 개선)
- **13:20** — HTML 시각화 결과 및 SAM2 캐시 추가

---

## Phase 5 — 실전 통합 및 최종 완성 (3/21 오후 14:00~15:30)

### 로봇 현장 통합 (woongje-cho)

- **14:08** — 컨트롤러 IP 및 카메라 해상도를 현장 환경에 맞게 업데이트
  - `CONTROLLER_IP` → 실제 로봇 IP로 변경
- **14:10** — `WJ/` 디렉토리: 9개 pick-and-place 스크립트 추가 (총 2,697줄)
  - `groot_pick_and_place.py`: 메인 데모 (SAM2 + GPT-4o + 로봇 제어)
  - 24점 카메라-로봇 캘리브레이션 데이터 포함
  - RealSense → SAM2 → GPT-4o CoT → ROS2 servoj 전체 루프

### 음성 입력 통합 (JJW2602)

- **14:43** — faster-whisper 기반 STT 음성 입력을 pick-and-place에 통합
  - PyAudio 녹음 → Whisper 변환 → GPT-4o 추론 연결

### 태스크 관리 및 에러 복구 (jun)

- **14:37** — 태스크 분해 + 관계 기반 PLACE End-Effector 로직
- **15:19** — `task_manager.py` (245줄): 세션 히스토리 추적 + 액션 검증
- **15:19** — base_25 이미지, SAM2 캐시, 세션 히스토리, HTML 결과 추가
- **15:29** — **에러 복구 기능 완성**: 실패 진단 + EE 보정 후 재시도

---

## 기술 아키텍처 요약

```
음성/텍스트 입력
    ↓
[STT] faster-whisper (한국어 → 텍스트)
    ↓
[카메라] RealSense (1280×720 RGB-D)
    ↓
[세그멘테이션] SAM2-tiny (blind segmentation → 마스크)
    ↓
[추론] GPT-4o CoT (타겟 식별 + 파지점 + EE 계산)
    ↓
[캘리브레이션] 24점 카메라→로봇 좌표 변환
    ↓
[제어] ROS2 servoj (10Hz 서보 루프)
    ↓
[안전] 3-tier safety clamp (delta/position/velocity)
    ↓
[복구] 실패 감지 → 진단 → EE 보정 → 재시도
    ↓
Doosan E0509 (6-DOF + 그리퍼)
```

## 핵심 성과

| 항목 | 결과 |
|------|------|
| 파이프라인 속도 | 20분 → 53초 (23x 개선) |
| Graspability Precision | 50% → 75% (ICL 4-shot) |
| VLM 전환 | Gemini ER → GPT-4o (정확도 + 속도 향상) |
| 폴백 전략 | 3단계 (GR00T → SmolVLA → Classical) |
| 안전 시스템 | 3-tier clamp + 실패 감지 + 자동 복구 |
| 코드 규모 | Python ~12,000줄+, 테스트 20+개 |
| 개발 기간 | ~26시간 (3/19 저녁 ~ 3/21 오후) |

## 주요 파일 구조

```
groot/
├── configs/doosan_e0509_config.py    # 로봇/카메라/안전 설정
├── utils/
│   ├── gemini_bridge.py              # Gemini ER → 로봇 브릿지 (5단계 closed-loop)
│   ├── gemini_saycan.py              # ReAct 태스크 오케스트레이터
│   ├── object_localizer.py           # SAM2 + GPT-4o 파이프라인
│   ├── task_manager.py               # 세션 히스토리 + 액션 검증
│   ├── calibration.py                # Hand-Eye 캘리브레이션
│   └── rgbd_localizer.py             # RGBD 기반 로컬라이제이션
├── vla/
│   ├── doosan_vla_controller.py      # VLA 메인 제어 루프
│   ├── doosan_action_adapter.py      # 3-tier 안전 클램프
│   ├── failure_detector.py           # Stall/Over-clamp 감지
│   └── sim/                          # Isaac Sim 시뮬레이션
├── WJ/
│   └── groot_pick_and_place.py       # 최종 데모 (SAM2+GPT-4o+로봇)
├── tests/                            # 20+ 테스트 파일
├── scripts/                          # 5단계 실행 스크립트
└── results/                          # 19개 결과 디렉토리
```

---

*Generated: 2026-03-21*
