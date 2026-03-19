# 🤖 Physical AI 해커톤 — V100 실험 + 최종 아키텍처

> **해커톤**: 2026.03.20(금) 20시 ~ 03.21(토) 22시 (26시간)
> **로봇**: 두산 E-Series 6-DOF + gripper (7-dim action)
> **목표 태스크**: "저 긴 물체 좀 가져다줘" — 모호한 한국어 텍스트 명령으로 로봇이 추론하여 pick & place
> **모델**: GR00T N1.6 (3B VLA)
> **Action space**: Joint position relative delta [Δj1..Δj6, grip]

---

## 최종 추론 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                every timestep @ ~10Hz closed loop         │
└─────────────────────────────────────────────────────────┘

 ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐
 │  텍스트 명령       │  │  카메라 RGB      │  │  로봇 상태       │
 │ "저 긴 물체 좀     │  │  외부 + wrist    │  │ [j1..j6, grip]  │
 │  가져다줘"         │  │  (optional)      │  │  7-dim          │
 └────────┬─────────┘  └───────┬─────────┘  └───────┬────────┘
          │                     │                     │
          ▼                     ▼                     ▼
 ┌────────────────────────────────────────────────────────┐
 │  VLA model (GR00T N1.6)                                │
 │  ┌─────────────────────┐  ┌──────────────────────────┐ │
 │  │ VLM backbone         │  │ Embodiment-specific      │ │
 │  │ (frozen or LoRA)     │  │ state encoder MLP        │ │
 │  │ visual grounding     │  │ (두산 7-dim 전용)         │ │
 │  └──────────┬──────────┘  └────────────┬─────────────┘ │
 │             └──────── fusion ──────────┘               │
 │                        │                               │
 │             ┌──────────▼──────────┐                    │
 │             │ Action head (DiT)   │                    │
 │             │ flow matching       │                    │
 │             └──────────┬──────────┘                    │
 │                        │                               │
 │             ┌──────────▼──────────┐                    │
 │             │ Embodiment-specific │                    │
 │             │ action decoder MLP  │                    │
 │             └──────────┬──────────┘                    │
 └────────────────────────┼───────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Joint delta chunk     │
              │ [Δj1..Δj6, grip]     │
              │ × 8 steps @ 10Hz     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Action adapter        │
              │ + safety clamp        │
              │ current + Δ → target  │
              │ joint limit clip      │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ 두산 E-Series         │
              │ servoj 명령 전송       │
              │ 접근→파지→이동→전달    │
              └───────────┬───────────┘
                          │
                  ┌───────┴───────┐
                  │ feedback      │
                  │ 새 RGB image  │
                  │ 새 joint state│
                  └───────┬───────┘
                          │
                    (다음 timestep)
```

### 핵심 설계 결정

- **STT 제외**: 데모에서는 텍스트 명령을 직접 입력. "저 긴 물체 좀 가져다줘" 같은 모호한 명령을 모델이 추론
- **Joint delta action**: Cartesian EE delta가 아닌 joint position relative delta 사용
  - 이유 1: 두산은 VLA pretrain 데이터에 없어서 Cartesian convention 매핑이 불확실
  - 이유 2: 배포 시 IK 변환 불필요, `current_joint + Δ → servoj`로 직접 전송
  - 이유 3: GR00T N1.6이 state-relative action을 기본으로 사용하므로 호환
- **Action chunking**: 1회 inference로 8 step 예측, 10Hz 제어 → 0.8초 분량의 행동 계획
- **Embodiment-specific encoder/decoder**: GR00T의 핵심 설계. 두산을 새 embodiment로 등록하면 전용 MLP가 자동 학습

### 학습 파이프라인

```
실물 두산 로봇 텔레옵 (direct teaching / SpaceMouse)
    ↓
동기 녹화: RGB + joint_pos(7-dim) + language label (30~50개)
    ↓
[선택] Isaac Sim joint replay + visual DR (→ 300~500개 증폭)
    ↓
LeRobot v2 포맷 데이터셋 통합
    ↓
GR00T N1.6 LoRA fine-tuning (Option A: VLM frozen / Option B: VLM LoRA)
    ↓
실물 로봇 배포: inference server → action adapter → servoj
```

---

## 실험 0: 환경 셋업

### 0-1. CUDA / GPU 확인

```bash
nvidia-smi
# V100 16GB 또는 A100 40/80GB 확인
```

### 0-2. GR00T N1.6 설치

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# GR00T repo 클론
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# 의존성 설치
bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate

# 확인
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

### 0-3. 체크포인트 다운로드

```bash
pip install huggingface_hub
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ./checkpoints/groot_n1.6_3b
ls -lh checkpoints/groot_n1.6_3b/
```

**✅ 체크포인트**: torch.cuda.is_available() = True, 모델 파일 존재 확인

---

## 실험 1: Pretrained 모델 Inference

> 목적: 모델 로딩 + action 출력 확인
> 예상: 30분

### 1-1. 공식 inference 테스트

```bash
# 샘플 데이터가 있다면
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/groot_n1.6_3b \
    --dataset-path demo_data/robot_sim.PickNPlace \
    --embodiment-tag new_embodiment \
    --traj-ids 0 \
    --inference-mode pytorch \
    --action-horizon 8
```

### 1-2. 더미 입력 수동 테스트

```python
# test_inference.py
"""GR00T N1.6 inference 기본 테스트"""
import torch
import numpy as np
import time

# 모델 import — 에러 나면 설치 문제
try:
    from gr00t.model.policy import Gr00tPolicy
    print("✅ gr00t import 성공")
except ImportError as e:
    print(f"❌ import 실패: {e}")
    print("설치 확인: pip install -e . 또는 uv pip install -e .")
    exit(1)

# 모델 로드
print("모델 로딩 중...")
t0 = time.time()
policy = Gr00tPolicy(
    model_path="checkpoints/groot_n1.6_3b",
    embodiment_tag="new_embodiment",
    device="cuda"
)
print(f"✅ 모델 로딩 완료: {time.time()-t0:.1f}초")

# GPU 메모리 확인
mem_gb = torch.cuda.memory_allocated() / 1e9
print(f"GPU 메모리 사용: {mem_gb:.2f} GB")

# 더미 inference
dummy_input = {
    "video.ego_cam": np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
    "state.joint_position": np.zeros((1, 7), dtype=np.float32),
    "annotation.human.action.task_description": ["pick up the long object"],
}

print("Inference 실행 중...")
t0 = time.time()
with torch.no_grad():
    action = policy.get_action(dummy_input)
latency = (time.time() - t0) * 1000

print(f"✅ Action shape: {action.shape}")
print(f"✅ Action sample: {action[0][:7]}")
print(f"✅ Inference latency: {latency:.1f}ms")
print(f"✅ GPU 메모리 피크: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

```bash
python test_inference.py
```

### 기록할 것

```
모델 로딩 시간: ___초
GPU 메모리 (inference): ___GB
Action output shape: ___
1회 inference 시간: ___ms
→ 10Hz 제어 가능 여부: (100ms 이하이면 OK)
```

---

## 실험 2: 두산 Embodiment 등록

> 목적: 새 embodiment config 작성 + 모델 인식 확인
> 예상: 1시간

### 2-1. Config 파일 작성

```python
# doosan_config.py
"""
두산 E-Series 6-DOF + gripper embodiment 설정
해커톤 현장에서 실제 joint limits 확인 후 수정할 것
"""

DOOSAN_EMBODIMENT = {
    "embodiment_tag": "doosan_e_series",
    
    # 7-dim: 6 joints + 1 gripper
    "state_dim": 7,
    "action_dim": 7,
    
    # Joint limits (라디안) — M1013 기준 추정값, 현장에서 교체
    "joint_limits": [
        [-6.2832, 6.2832],   # j1
        [-6.2832, 6.2832],   # j2
        [-2.7925, 2.7925],   # j3
        [-6.2832, 6.2832],   # j4
        [-6.2832, 6.2832],   # j5
        [-6.2832, 6.2832],   # j6
        [0.0, 1.0],          # gripper (0=open, 1=close)
    ],
    
    # Action = joint position relative delta
    "action_type": "joint_delta_relative",
    "action_horizon": 8,
    "control_hz": 10,
    
    # Camera
    "cameras": ["ego_cam"],  # 외부 카메라 1대, wrist 추가 시 ["ego_cam", "wrist_cam"]
    "image_size": [224, 224],
}

print("Doosan embodiment config:")
for k, v in DOOSAN_EMBODIMENT.items():
    print(f"  {k}: {v}")
```

### 2-2. GR00T의 finetune_new_embodiment 가이드 확인

```bash
# 공식 가이드 읽기
cat getting_started/finetune_new_embodiment.md

# modality config 예제 확인
ls getting_started/
cat getting_started/data_preparation.md
```

### 2-3. Modality config JSON 작성

```bash
# 실제 GR00T가 요구하는 포맷에 맞춰 작성
# getting_started/finetune_new_embodiment.md의 예제를 참고하여 수정
# → 이 부분은 가이드 읽은 후 구체적으로 작성
```

**✅ 체크포인트**: embodiment_tag="doosan_e_series"로 inference 호출 시 에러 없이 action 출력

---

## 실험 3: 더미 데이터로 Fine-tuning 파이프라인 검증

> 목적: SFT가 실제로 돌아가는지 + 1 step 시간 측정
> 예상: 1~2시간 (이게 제일 중요한 실험)

### 3-1. 더미 데이터셋 생성

```python
# create_dummy_dataset.py
"""
해커톤에서 실제 두산 로봇 데모로 교체할 더미 데이터셋
LeRobot v2 호환 포맷
"""
import numpy as np
import json
import os

DATASET_DIR = "dummy_doosan_data"
N_EPISODES = 5       # 해커톤에서는 30~50
EPISODE_LEN = 50     # 5초 분량 @ 10Hz
ACTION_DIM = 7       # 6 joints + gripper

os.makedirs(DATASET_DIR, exist_ok=True)

all_episodes = []

for ep in range(N_EPISODES):
    # 더미 joint trajectory (실제로는 로봇 녹화 데이터)
    joints = np.cumsum(
        np.random.randn(EPISODE_LEN, ACTION_DIM) * 0.01, axis=0
    ).astype(np.float32)
    
    # gripper: 0.5 이전은 open, 이후는 close (pick 시뮬레이션)
    joints[:, 6] = 0.0
    joints[EPISODE_LEN//2:, 6] = 1.0
    
    episode = {
        "state": joints,                                    # (T, 7)
        "action": np.diff(joints, axis=0, prepend=joints[:1]),  # delta
        "images": np.random.randint(0, 255, (EPISODE_LEN, 224, 224, 3), dtype=np.uint8),
        "language": "pick up the long object",
    }
    all_episodes.append(episode)
    print(f"  Episode {ep+1}/{N_EPISODES}: {EPISODE_LEN} steps")

np.savez_compressed(
    f"{DATASET_DIR}/dummy_episodes.npz",
    episodes=all_episodes
)
print(f"\n✅ 저장 완료: {DATASET_DIR}/dummy_episodes.npz")
print(f"   {N_EPISODES} episodes × {EPISODE_LEN} steps = {N_EPISODES * EPISODE_LEN} samples")
print(f"   Action dim: {ACTION_DIM} (6 joints + gripper)")
print(f"   Action type: joint position relative delta")
```

```bash
python create_dummy_dataset.py
```

### 3-2. LeRobot v2 포맷 변환

```bash
# GR00T의 데이터 변환 스크립트 사용
# getting_started/data_preparation.md 참고하여 실행
# 정확한 명령어는 가이드 읽은 후 결정

# 예상 명령어 (가이드 확인 후 수정):
uv run python getting_started/data_preparation.py \
    --input-path dummy_doosan_data/ \
    --output-path dummy_doosan_lerobot/ \
    --embodiment-tag doosan_e_series
```

### 3-3. Fine-tuning 실행 (100 steps만)

```bash
# ⚠️ V100 16GB에서는 메모리 주의
# batch-size 1, lora-rank 8로 시작

uv run python scripts/gr00t_finetune.py \
    --dataset-path dummy_doosan_lerobot/ \
    --model-path checkpoints/groot_n1.6_3b \
    --embodiment-tag doosan_e_series \
    --output-dir test_checkpoint/ \
    --num-gpus 1 \
    --max-steps 100 \
    --batch-size 1 \
    --lora-rank 16 \
    --learning-rate 5e-5

# V100에서 OOM 발생 시 시도할 옵션들 (순서대로):
# 1. --lora-rank 8
# 2. --no-tune_diffusion_model (action head만 학습, VLM frozen)  
# 3. --batch-size 1 (이미 최소)
# A100이면 --batch-size 4 --lora-rank 32 가능
```

### 3-4. 결과 기록 (⭐ 가장 중요)

```
Fine-tuning 시작 성공 여부: ☐ Yes / ☐ No
1 step 소요 시간: ___초
GPU 메모리 피크: ___GB
100 steps 총 소요 시간: ___분

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
해커톤 학습 시간 추정:
  데모 50개 × 50 timestep = 2500 samples
  목표: 10K~20K steps
  
  예상 시간 = (1 step ___초) × 20000 / 3600 = ___시간
  
  3090×8이면: ___시간 / 8 ≈ ___시간 (DDP)
  4060×1이면: 위 시간 그대로, VRAM 8GB → OOM 가능성 높음
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 실험 4: Fine-tuned 체크포인트 Inference

> 목적: 학습된 모델 로딩 + action 출력 확인
> 예상: 15분

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path test_checkpoint/ \
    --embodiment-tag doosan_e_series \
    --dataset-path dummy_doosan_lerobot/ \
    --traj-ids 0 \
    --inference-mode pytorch \
    --action-horizon 8
```

**✅ 체크포인트**: fine-tuned 모델에서 7-dim action chunk 정상 출력

---

## 실험 5: VLM 오케스트레이터 (Plan B)

> 목적: VLA가 실패할 경우의 fallback 파이프라인
> GPU 불필요, API만 사용

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

```python
# test_vlm_plan_b.py
"""
Plan B: VLM으로 물체 인식 → classical IK controller로 pick & place
VLA fine-tuning이 시간 내에 안 끝나거나 성능이 부족할 때의 fallback
"""
import anthropic
import base64
import json
import sys

client = anthropic.Anthropic()

def identify_target(image_path: str, command: str) -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": f"""Robot manipulation assistant.
User command: "{command}"
Identify the target object. Estimate bounding box (normalized 0-1).
JSON only:
{{"target": "...", "reasoning": "...", "bbox": {{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}}, "action": "pick_and_handover|pick_and_place_in_bin", "confidence": ...}}"""}
            ],
        }],
    )
    return json.loads(response.content[0].text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vlm_plan_b.py <image.jpg>")
        print("테이블 위 물체 이미지 아무거나 다운로드해서 테스트")
        sys.exit(1)
    
    commands = [
        "저 긴 물체 좀 가져다줘",
        "빨간색 물건을 통에 넣어",
        "오른쪽에 있는 거 집어줘",
    ]
    
    for cmd in commands:
        print(f"\n{'='*50}")
        print(f"Command: {cmd}")
        print('='*50)
        result = identify_target(sys.argv[1], cmd)
        print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 실험 6: 배포 코드 (해커톤 현장용 스켈레톤)

### 6-1. Action Adapter

```python
# doosan_action_adapter.py
"""
VLA joint delta → 두산 로봇 servoj 명령 변환
해커톤 현장에서 joint_limits를 실제 값으로 교체
"""
import numpy as np

class DoosanActionAdapter:
    def __init__(self):
        # 두산 M1013 joint limits — 현장에서 확인 후 교체!
        self.joint_limits = np.array([
            [-6.28, 6.28], [-6.28, 6.28], [-2.79, 2.79],
            [-6.28, 6.28], [-6.28, 6.28], [-6.28, 6.28],
        ])
        self.max_delta = 0.05  # rad, safety limit per step
        self.gripper_threshold = 0.5
    
    def convert(self, vla_action: np.ndarray, current_joints: np.ndarray) -> dict:
        """
        vla_action: (7,) [Δj1..Δj6, grip]
        current_joints: (6,) from robot
        """
        delta = np.clip(vla_action[:6], -self.max_delta, self.max_delta)
        target = current_joints + delta
        
        for i in range(6):
            target[i] = np.clip(target[i], self.joint_limits[i,0], self.joint_limits[i,1])
        
        return {
            "joint_target": target,
            "gripper_close": vla_action[6] > self.gripper_threshold,
        }

if __name__ == "__main__":
    adapter = DoosanActionAdapter()
    fake_action = np.array([0.01, -0.02, 0.015, 0.0, 0.005, -0.01, 0.8])
    current = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
    result = adapter.convert(fake_action, current)
    print(f"Target: {result['joint_target']}")
    print(f"Gripper close: {result['gripper_close']}")
```

### 6-2. Inference Server 실행 (해커톤용)

```bash
# GPU 서버에서 실행 (3090 or V100)
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag doosan_e_series \
    --model-path test_checkpoint/ \
    --port 8000
```

### 6-3. ROS2 Controller 스켈레톤 (현장 전용)

```python
# doosan_vla_controller.py
"""
해커톤 현장에서만 실행 — ROS2 + 실물 두산 로봇 필요
"""

# === 현장에서 uncomment ===
# import rclpy
# from sensor_msgs.msg import JointState, Image
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class DoosanVLAController:
    """
    Control loop:
    1. 카메라 RGB 수신 (ROS2 /camera/color/image_raw)
    2. Joint state 수신 (ROS2 /joint_states)
    3. GR00T inference server에 전송 (websocket :8000)
    4. Action adapter + safety clamp
    5. servoj로 두산 로봇 제어
    """
    def __init__(self, language_instruction="pick up the long object"):
        self.instruction = language_instruction
        self.rate_hz = 10
        print(f"Controller ready. Instruction: '{self.instruction}'")
        print(f"Control rate: {self.rate_hz}Hz")
        print("⚠️  ROS2 + 두산 로봇 연결 필요 — 해커톤 현장에서 실행")

if __name__ == "__main__":
    ctrl = DoosanVLAController("저 긴 물체 좀 가져다줘")
```

---

## 실험 결과 요약 템플릿

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 실험 결과 (V100에서 측정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[실험 0] 환경 셋업
  GPU: ___
  CUDA: ___
  GR00T 설치: ☐ OK / ☐ 실패 (원인: ___)

[실험 1] Inference
  로딩 시간: ___초
  VRAM 사용: ___GB
  Action shape: ___
  Latency: ___ms → 10Hz 가능: ☐ Yes / ☐ No

[실험 2] Embodiment 등록
  doosan_e_series 등록: ☐ OK / ☐ 실패 (원인: ___)

[실험 3] Fine-tuning ⭐
  1 step 시간: ___초
  VRAM 피크: ___GB
  100 steps 소요: ___분
  해커톤 20K steps 추정: ___시간
  → 26시간 내 가능: ☐ Yes / ☐ No

[실험 4] Fine-tuned inference
  체크포인트 로딩: ☐ OK / ☐ 실패
  7-dim action 출력: ☐ OK / ☐ 실패

[실험 5] VLM Plan B
  grounding 정확도: ___/3 (3개 명령 중)
  
[실험 6] 배포 코드
  Action adapter: ☐ 작성 완료
  ROS2 스켈레톤: ☐ 작성 완료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

해커톤 가져갈 것:
☐ Isaac-GR00T/ (repo + checkpoint)
☐ doosan_config.py
☐ doosan_action_adapter.py
☐ doosan_vla_controller.py (skeleton)
☐ test_vlm_plan_b.py
☐ 위 측정값 메모
☐ USB 백업 (체크포인트 + 코드)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
