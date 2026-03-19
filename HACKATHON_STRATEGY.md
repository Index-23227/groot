# Physical AI 해커톤 전략 + 현황

> **해커톤**: 2026.03.20(금) 20시 ~ 03.21(토) 22시 (26시간)
> **로봇**: 두산 E0509 (6-DOF + gripper)
> **모델**: GR00T N1.6 3B VLA
> **GPU (현장)**: 3090 × 8

---

## 현재 완료된 것 (V100 사전 테스트)

### 코드 (Isaac-GR00T/examples/doosan/)

| 파일 | 설명 | 상태 |
|------|------|------|
| `doosan_config.py` | Embodiment modality 설정 (6-DOF + gripper) | 완료, 검증됨 |
| `convert_teleop_to_lerobot.py` | 텔레옵 CSV/JSONL+비디오 → LeRobot v2 변환기 | 완료, end-to-end 테스트 통과 |
| `launch_server.py` | 추론 서버 (server/client/local/benchmark) | 완료, V100 벤치마크 완료 |
| `doosan_action_adapter.py` | VLA 출력 → servoj 명령 (safety clamp 포함) | 완료 |
| `doosan_vla_controller.py` | ROS2 제어 루프 스켈레톤 | 스켈레톤 완료, 현장에서 연결 |
| `create_dummy_dataset.py` | 더미 데이터셋 생성 | 완료 |
| `run_finetune.sh` | Fine-tuning 실행 스크립트 | 완료, V100 전용 |
| `test_inference.py` | 추론 테스트 (processor monkey-patching) | 완료, 검증됨 |
| `test_vlm_plan_b.py` | VLM fallback (Claude API) | 작성 완료, API key 필요 |
| `README.md` | 3090 마이그레이션 가이드 | 완료 |

### V100 실험 결과

| 항목 | 결과 |
|------|------|
| Fine-tuning 1 step | ~5.66초 (batch=1) |
| Fine-tuning VRAM 피크 | ~14GB / 16GB |
| 100 steps 소요 | 9분 26초 |
| Train loss | 1.35 → 1.25 (학습 진행 확인) |
| Inference VRAM | 5.4 GB |
| Inference latency | ~500ms (eager attn) |
| Inference 처리량 | ~2 Hz |

### V100 전용 패치 (3090에서 되돌려야 함)

**자세한 내용은 `examples/doosan/README.md` 섹션 1-3 참고**

핵심:
- `config.json`: select_layer 8→16, tune_top_llm_layers 0→4 복원
- optimizer: adafactor → adamw_torch
- dtype: fp16 → bf16, tf32 활성화
- backbone CPU offload 제거
- video_backend: opencv → torchcodec

---

## 해커톤 당일 타임라인

### Phase 0: 환경 셋업 (20:00 ~ 20:30)

```bash
# 1. 레포 클론
git clone https://github.com/Index-23227/Isaac-GR00T.git
cd Isaac-GR00T
pip install -e .

# 2. 체크포인트 다운로드
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ./checkpoints/groot_n1.6_3b

# 3. V100 패치 되돌리기 (README.md 섹션 1-3 참고)
cp checkpoints/groot_n1.6_3b/config.json.bak checkpoints/groot_n1.6_3b/config.json
sed -i 's/optim = "adafactor"/optim = "adamw_torch"/' gr00t/experiment/launch_finetune.py
# 나머지 패치: git diff 확인 후 revert
```

### Phase 1: 로봇 연결 + 데모 수집 (20:30 ~ 22:00)

```
1. 두산 E0509 연결 확인 (IP, ROS2, joint limits)
2. 카메라 설치 + 캘리브레이션
3. Direct Teaching으로 30~50 에피소드 수집
   - "pick up the [object]" 류 단순 태스크
   - 에피소드당 ~5초 (50 frames @ 10Hz)
   - CSV + 비디오 동시 녹화
```

### Phase 2: 데이터 변환 + Fine-tuning (22:00 ~ 01:00)

```bash
# 데이터 변환 (5분)
python examples/doosan/convert_teleop_to_lerobot.py \
    --input_dir /path/to/recordings \
    --output_dir /path/to/dataset \
    --task "pick up the object" \
    --fps 10

# Fine-tuning 시작 (3090×8: ~1.5시간)
python gr00t/experiment/launch_finetune.py \
    --base_model_path checkpoints/groot_n1.6_3b \
    --dataset_path /path/to/dataset \
    --modality_config_path examples/doosan/doosan_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus 8 \
    --max_steps 20000 \
    --global_batch_size 256 \
    --output_dir /path/to/output
```

### Phase 3: 추론 서버 + 로봇 테스트 (01:00 ~ 04:00)

```bash
# 서버 실행
python examples/doosan/launch_server.py \
    --mode server \
    --model_path /path/to/output/checkpoint-20000

# 로봇 제어 (별도 터미널)
# doosan_vla_controller.py 수정 후 실행
```

### Phase 4: 반복 개선 (04:00 ~ 18:00)

```
- 실패 케이스 분석 → 추가 데모 수집 → 재학습
- Action adapter 안전 파라미터 튜닝
- 데모 시나리오 준비
```

### Phase 5: 발표 준비 (18:00 ~ 22:00)

```
- 데모 영상 녹화
- 발표 자료 (아키텍처 다이어그램, 실험 결과)
- 라이브 데모 리허설
```

---

## 아키텍처

```
[카메라 RGB] + [Joint State] + [언어 명령]
                    │
                    ▼
         ┌─────────────────────┐
         │  GR00T N1.6 (3B)    │
         │  Eagle VLM backbone │
         │  + DiT action head  │
         └──────────┬──────────┘
                    │
          Joint delta chunk (8 steps)
          [Δj1..Δj6, grip] × 8
                    │
                    ▼
         ┌─────────────────────┐
         │  Action Adapter     │
         │  safety clamp       │
         │  current + Δ → target│
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  두산 E0509 servoj  │
         └─────────────────────┘
```

통신: 로봇 PC ←ZMQ→ GPU 서버 (PolicyClient/PolicyServer)

---

## Plan B: VLM fallback

VLA fine-tuning이 실패하거나 성능 부족 시:

```
카메라 → Claude Vision API → 물체 인식 + bbox
    → Classical IK planner → servoj
```

스크립트: `examples/doosan/test_vlm_plan_b.py` (ANTHROPIC_API_KEY 필요)

---

## 리스크 + 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| 3090 OOM | 낮음 (24GB) | batch size 줄이기, gradient accumulation |
| 학습 시간 초과 | 중간 | 5K steps만 (3090×8: ~20분), 나중에 추가 |
| 추론 너무 느림 | 낮음 | Action horizon 8 → 실질 커버 OK |
| 로봇 통신 문제 | 중간 | DRCF TCP 직접 통신 fallback |
| 텔레옵 데이터 품질 | 높음 | 에피소드 많이 수집, 불량 제거 |
| VLA 성능 부족 | 중간 | Plan B (VLM + IK) |

---

## 파일 위치 정리

```
/scratch/x3326a25/groot/
├── HACKATHON_STRATEGY.md          # 이 파일 (전략 + 타임라인)
├── experiment_results.md           # V100 실험 측정값
├── Isaac-GR00T/                    # GR00T 레포 (fork: Index-23227)
│   ├── examples/doosan/            # 두산 전용 코드 전부
│   │   ├── README.md               # 3090 마이그레이션 가이드
│   │   ├── convert_teleop_to_lerobot.py
│   │   ├── launch_server.py
│   │   ├── doosan_config.py
│   │   ├── doosan_action_adapter.py
│   │   ├── doosan_vla_controller.py
│   │   └── ...
│   └── checkpoints/groot_n1.6_3b/  # 모델 체크포인트
└── (구 파일들 - 레포에 통합됨)
    ├── hackathon_groot_v100_guide.md
    ├── hackathon_pre_prep.md
    ├── doosan_action_adapter.py
    └── doosan_m0608_config.py
```

**핵심 코드는 전부 `Isaac-GR00T/examples/doosan/`에 있음.**
**구 파일들은 레포에 통합되었으므로 참고용.**
