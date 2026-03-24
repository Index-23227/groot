# 전략 변천사: GR00T VLA에서 SAM2+GPT-4o까지

> 26시간 해커톤 동안 계획이 어떻게, 왜 바뀌었는가

---

## 원래 계획: GR00T VLA Fine-tuning

해커톤 시작 전, 팀의 전략은 명확했다:

```
텔레옵 데이터 수집 → GR00T N1.6 Fine-tuning → End-to-End VLA 제어
```

- **모델**: NVIDIA GR00T N1.6 3B (Eagle VLM backbone + DiT action head)
- **GPU**: 3090 × 8 (현장 제공)
- **목표**: 카메라 RGB + Joint State + 언어 명령 → Joint delta 직접 출력
- **Plan B**: VLM(Claude Vision) + Classical IK planner

V100에서 사전 테스트까지 완료한 상태였다 (Fine-tuning 100 steps 9분 26초, Inference ~500ms).

**이 계획의 핵심 가정**: 현장에서 30~50 에피소드 텔레옵 데이터를 빠르게 수집하고, 3090×8로 1.5시간 내 Fine-tuning을 끝낼 수 있다.

---

## 전환점 1: 모듈화와 3단 폴백 (3/20 오전)

### 무슨 일이 있었나

해커톤이 시작되고, 코드를 정리하면서 현실적인 리스크가 보이기 시작했다:
- 텔레옵 데이터 수집 품질이 불확실
- VLA Fine-tuning 시간이 예상보다 길어질 수 있음
- 로봇 현장 환경이 테스트 환경과 다를 수 있음

### 어떻게 바뀌었나

코드를 모듈 구조(`configs/`, `utils/`, `scripts/`, `vla/`)로 재조직하면서, 3단계 폴백 전략을 수립:

```
Level 1: GR00T VLA (End-to-End)     ← 메인
Level 2: SmolVLA (경량 대안)        ← 폴백 1
Level 3: VLM + Classical Planner    ← 폴백 2 (Plan C)
```

STT 음성 입력, 실패 감지기, Isaac Sim 시뮬레이션 등 "보험" 모듈도 이때 추가.

### 왜?

> 26시간은 VLA Fine-tuning 한 번 실패하면 복구할 시간이 없다. 보험이 필요했다.

---

## 전환점 2: Gemini ER 진입 (3/20 밤 ~ 3/21 새벽)

### 무슨 일이 있었나

3/20 밤 23:11, `track2-gemini-curobo.tar.gz`가 업로드된다. Track 2용 Gemini+cuRobo 코드. 이것이 방향 전환의 첫 신호였다.

3/21 새벽 02:25, Gemini Embodied Reasoning(ER) 1.5 테스트가 시작된다. 자동 평가 파이프라인까지 갖춘 본격적 실험이었다.

### 어떻게 바뀌었나

```
GR00T VLA (End-to-End 학습)
    ↓ 사실상 포기
Gemini ER (VLM으로 물체 인식 + 추론) + Classical 제어
```

- `gemini_bridge.py`: Gemini ER → 로봇 실행 5단계 closed-loop
- `gemini_saycan.py`: SayCan 스타일 ReAct 패턴 오케스트레이터
- Graspability(파지 가능성) 평가: ICL 4-shot으로 Precision 50% → 75%

### 왜?

> GR00T VLA는 "학습 → 추론"의 파이프라인이 무겁다. 텔레옵 데이터 수집, 변환, Fine-tuning, 디버깅... 26시간 안에 이 모든 걸 안정적으로 끝내기 어려웠다. Gemini ER은 학습 없이 바로 추론이 가능했다.

---

## 전환점 3: SAM2 도입과 파이프라인 최적화 (3/21 04:30~06:40)

### 무슨 일이 있었나

Gemini ER만으로는 물체 위치를 정밀하게 잡기 어려웠다. Bounding box는 나오지만, 3D 좌표로의 변환이 부정확했다.

### 어떻게 바뀌었나

```
카메라 RGB
    ↓
SAM2-tiny (blind segmentation → 마스크)
    ↓
VLM (마스크 중 타겟 식별)
    ↓
Depth → 3D 좌표
    ↓
로봇 제어
```

SAM2가 모든 물체를 먼저 세그멘테이션하고, VLM이 "이 중에서 어떤 게 타겟인가"를 판단하는 2단계 구조.

초기에는 **20분**이나 걸렸지만, 세 가지 최적화로 **53초**까지 단축:
1. SAM2 마스크 캐싱 (같은 장면 재계산 방지)
2. Top-5 필터 (너무 작은 마스크 제거)
3. Parallel VLM 호출 (마스크별 동시 추론)

### 왜?

> VLM은 "이해"는 잘하지만 "위치 특정"은 못한다. SAM2는 "위치"는 잘 잡지만 "이해"는 못한다. 둘의 조합이 답이었다.

---

## 전환점 4: Gemini → GPT-4o 교체 (3/21 08:11) ⭐

### 무슨 일이 있었나

이것이 프로젝트의 **가장 큰 전환점**이다. 08:11 커밋 메시지: "Replace Gemini ER with GPT-4o, add SAM2-free RGBD pipeline."

### 어떻게 바뀌었나

```
SAM2 + Gemini ER  →  SAM2 + GPT-4o
```

동시에 `rgbd_localizer.py`(346줄)라는 SAM2 없이 RGBD만으로 동작하는 대안 파이프라인도 추가.

### 왜?

커밋 기록에서 직접적인 이유가 명시되진 않지만, 맥락에서 추론할 수 있다:

1. **정확도**: GPT-4o의 시각 추론이 Gemini ER보다 일관성 있었을 가능성
2. **속도**: API 응답 시간 차이
3. **CoT(Chain-of-Thought)**: 이후 모든 파이프라인이 CoT 기반으로 전환됨 — GPT-4o의 구조화된 추론이 더 적합했을 수 있음

> 핵심: "VLM은 도구다. 더 잘 맞는 도구로 바꿀 수 있는 구조를 만들어뒀기에 교체가 가능했다."

---

## 전환점 5: CoT 확정과 Pipeline A/B 비교 (3/21 09:00~13:00)

### 무슨 일이 있었나

GPT-4o 교체 후, 두 가지 접근법을 정식으로 비교 실험:
- **Pipeline A**: SAM2 + GPT-4o CoT
- **Pipeline B**: Depth-only 클러스터링

### 어떻게 바뀌었나

Pipeline A(SAM2 + CoT)가 승리. 이후 CoT 방식이 계속 진화:

```
09:04  Pipeline A/B 비교 → A 선택
11:14  SAM2 + CoT (ROI 기반 세그멘테이션)
12:13  Pick-and-Place CoT with geometric EE 계산
13:19  Scene-object-free CoT (일반화 개선)
```

CoT가 단순 "물체 찾기"에서 "파지점 계산 + End-Effector 자세 결정"까지 확장.

### 핵심 알고리즘: 윗면 EE 계산 (`compute_top_surface_ee`)

VLM이 타겟을 식별한 후, 실제 파지점(End-Effector 위치)은 **파라미터 없이 기하학적으로** 계산한다:

```
SAM2 마스크 + Depth 이미지
         ↓
    ┌────┴────┐
    ▼         ▼
 Point A    Point B
(depth 최소) (y 최소)
    └────┬────┘
         ▼
   EE = midpoint(A, B)
```

**Point A — "카메라에 가장 가까운 점"** (depth 최소):
```python
d_min = depths.min()
near_th = d_min + max((depths.max() - d_min) * 0.05, 3)  # 상위 5% or 최소 3mm
near = depths <= near_th
pt_a = centroid(xs[near], ys[near])
```
마스크 내 depth 값이 가장 작은(카메라에 가장 가까운) 상위 5% 픽셀 그룹의 centroid. 단일 최솟값 대신 그룹 centroid를 쓰는 이유는 센서 노이즈에 의한 불안정을 방지하기 위함.

**Point B — "이미지에서 가장 위쪽 점"** (y 최소):
```python
y_min = ys.min()
y_th = y_min + max((ys.max() - y_min) * 0.05, 3)  # 상위 5% or 최소 3px
top_y = ys <= y_th
pt_b = centroid(xs[top_y], ys[top_y])
```
마스크 내 y좌표가 가장 작은(이미지 상단 = 물체 윗면) 상위 5% 픽셀 그룹의 centroid.

**왜 두 점의 중점인가?**
- Point A(depth 최소)만 쓰면: 물체 앞면 중앙을 잡게 될 수 있음 (옆면을 집으려 함)
- Point B(y 최소)만 쓰면: 물체 윗면 가장자리를 잡게 될 수 있음 (불안정)
- 중점: 물체 윗면의 안정적인 중심부 — 위에서 내려잡기에 최적

이 방식의 장점은 **학습이나 하이퍼파라미터 튜닝 없이** SAM2 마스크와 depth 이미지만으로 파지점을 결정한다는 것.

### 왜?

> 구조화된 추론(CoT)이 단순 bbox 검출보다 로봇 조작에 필요한 정보를 더 풍부하게 제공했다. "어디를 잡을까"뿐 아니라 "어떤 각도로, 어떤 순서로"까지. 그리고 실제 파지점은 VLM에 의존하지 않고, 기하학적 계산으로 정밀하게 결정했다.

---

## 전환점 6: 현장 통합과 최종 시스템 (3/21 14:00~15:30)

### 무슨 일이 있었나

모든 기술적 결정이 끝나고, 실제 로봇과 연결하는 단계.

### 최종 시스템

```
음성(한국어) → faster-whisper STT
    ↓
RealSense 1280×720 RGB-D
    ↓
SAM2-tiny blind segmentation
    ↓
GPT-4o CoT (타겟 식별 + 파지점 + EE 계산)
    ↓
24점 Hand-Eye 캘리브레이션 (카메라 → 로봇 좌표)
    ↓
ROS2 servoj 10Hz + 3-tier safety clamp
    ↓
에러 복구 (실패 진단 → EE 보정 → 재시도)
    ↓
Doosan E0509
```

---

## 전략 변천 요약

```
시간        계획                    전환 이유
────────────────────────────────────────────────────────
사전 준비    GR00T VLA Fine-tuning   (원래 계획)
3/20 오전    + 3단 폴백 전략 수립     리스크 헤지
3/20 밤      Gemini ER 진입           VLA 학습 파이프라인이 무거움
3/21 04:30   + SAM2 도입              VLM만으로는 위치 특정 부정확
3/21 06:40   파이프라인 20분→53초     실용성 확보
3/21 08:11   Gemini → GPT-4o          정확도/CoT 적합성
3/21 09~13   CoT 확정 + 고도화        구조화된 추론이 로봇에 적합
3/21 14~15   현장 통합 + 완성          최종 시스템
```

## 교훈

1. **폴백을 준비하되, 빠르게 전환하라** — 3단 폴백 전략을 세웠지만, 실제로는 Plan C(VLM + Classical)의 발전형이 최종 답이 되었다.

2. **모듈화가 전환을 가능하게 한다** — VLM을 교체 가능한 컴포넌트로 설계했기에 Gemini → GPT-4o 전환이 한 커밋으로 끝났다.

3. **End-to-End보다 조합이 강했다** — GR00T VLA는 하나의 모델이 모든 걸 하려 했지만, SAM2(세그멘테이션) + GPT-4o(추론) + Classical(제어)의 조합이 26시간 해커톤에서는 더 실용적이었다.

4. **A/B 테스트는 시간 낭비가 아니다** — Pipeline A/B 비교에 시간을 쓴 덕분에 확신을 갖고 후속 개발에 집중할 수 있었다.

5. **최적화 타이밍이 중요하다** — 20분→53초 최적화가 없었다면, 이후의 모든 반복 실험이 불가능했을 것이다.

---

*이 문서는 git 커밋 히스토리(60개 커밋)와 기존 PROJECT_JOURNEY.md를 기반으로 재구성되었습니다.*
