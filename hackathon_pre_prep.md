# 🔧 해커톤 사전 준비: Embodiment 등록 + Action Adapter

> **목표**: 로봇 실물 없이 미리 준비할 수 있는 코드를 전부 작성해둔다.
> **현장에서 할 일**: joint limits 실측 확인 → config 숫자만 수정 → 통신 연결

---

## 1. 두산 E0509 스펙 정리

### Joint Specifications (공식 매뉴얼 기반)

| Joint | Range | Max Speed | Safety Range (권장) |
|-------|-------|-----------|-------------------|
| J1 | ±360° | 120°/s | ±350° |
| J2 | ±360° | 120°/s | ±95° (바닥 설치 시) |
| J3 | ±150° | 150°/s | ±145° |
| J4 | ±360° | 225°/s | ±350° |
| J5 | ±360° | 225°/s | ±350° |
| J6 | ±360° | 225°/s | ±350° |

> ⚠️ **주의**: E0509는 M0609와 기구학 구조가 거의 동일하나 속도가 약간 다름.
> J2 ±95°는 바닥 설치 시 자기충돌 방지 권장값. 현장에서 실제 설정 확인 필수.

### 기타 스펙
- **Payload**: 5 kg
- **Reach**: 900 mm
- **Repeatability**: ±0.05 mm
- **DOF**: 6 + gripper
- **Protection**: IP66 (F&B 전용)
- **Controller IP (default)**: 192.168.127.100
- **Controller Port (default)**: 12345

### ROS2 Joint Names (doosan-robot2 패키지)
```
joint1, joint2, joint3, joint4, joint5, joint6
```

---

## 2. GR00T N1.6 — Embodiment Config 사전 작성

### `doosan_e0509_config.py`

```python
"""
Doosan E0509 Embodiment Configuration for GR00T N1.6
미리 작성 가능 — 현장에서 joint limits만 실측 확인 후 업데이트
"""

import numpy as np

# ============================================================
# Doosan E0509 Embodiment Registration
# ============================================================

DOOSAN_E0509_CONFIG = {
    "embodiment_name": "doosan_e0509",
    "num_joints": 6,
    "num_actions": 7,  # 6 joints + 1 gripper
    
    # --- Joint Names (ROS2 doosan-robot2 convention) ---
    "joint_names": [
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6",
    ],
    
    # --- Joint Limits (radians) ---
    # ⚠️ 현장 실측 후 업데이트할 것
    "joint_limits_lower": np.deg2rad([-350, -95, -145, -350, -350, -350]),
    "joint_limits_upper": np.deg2rad([+350, +95, +145, +350, +350, +350]),
    
    # --- Joint Max Velocities (rad/s) ---
    "joint_max_velocities": np.deg2rad([120, 120, 150, 225, 225, 225]),
    
    # --- Gripper ---
    "gripper_range": [0.0, 1.0],  # 0=closed, 1=open (normalize at adapter)
    
    # --- Action Space ---
    "action_type": "joint_delta",  # state-relative delta
    "action_dim": 7,               # [Δj1, Δj2, Δj3, Δj4, Δj5, Δj6, grip]
    
    # --- Observation Space ---
    "state_dim": 7,                # [j1, j2, j3, j4, j5, j6, grip]
    
    # --- Control ---
    "control_frequency_hz": 10,    # target Hz (두산 E0509: 5~15Hz 가능)
    "action_horizon": 16,          # GR00T default chunk size
}

# ============================================================
# Normalization Statistics (placeholder — 데모 수집 후 계산)
# ============================================================

DOOSAN_NORM_STATS = {
    # 학습 전에 데모 데이터에서 mean/std 계산해서 채워넣기
    "action_mean": np.zeros(7),
    "action_std": np.ones(7),
    "state_mean": np.zeros(7),
    "state_std": np.ones(7),
}


def get_doosan_config():
    """GR00T N1.6 embodiment 등록 시 호출"""
    return DOOSAN_E0509_CONFIG


def compute_norm_stats(demo_dataset):
    """
    데모 수집 후 호출하여 normalization stats 계산.
    
    Args:
        demo_dataset: list of episodes, 각 episode는 dict with 'actions', 'states'
    Returns:
        dict with action_mean, action_std, state_mean, state_std
    """
    all_actions = np.concatenate([ep['actions'] for ep in demo_dataset], axis=0)
    all_states = np.concatenate([ep['states'] for ep in demo_dataset], axis=0)
    
    return {
        "action_mean": all_actions.mean(axis=0),
        "action_std": all_actions.std(axis=0).clip(min=1e-6),
        "state_mean": all_states.mean(axis=0),
        "state_std": all_states.std(axis=0).clip(min=1e-6),
    }
```

---

## 3. OpenPI (π₀.5) — Embodiment Config 사전 작성

### `doosan_e0509_openpi_config.py`

```python
"""
Doosan E0509 config for OpenPI (π₀ / π₀.5) fine-tuning.
openpi/src/openpi/training/config.py에 추가할 TrainConfig.
"""

# ============================================================
# OpenPI config.py에 추가할 코드
# ============================================================

"""
# --- 아래를 openpi/src/openpi/training/config.py의 _CONFIGS 리스트에 추가 ---

TrainConfig(
    name="pi05_doosan_e0509",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,       # 6 joints + 1 gripper
        action_horizon=16,  # action chunk length
        max_token_len=180,
    ),
    data=LeRobotDataConfig(
        repo_id="local/doosan_e0509_demos",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="./assets/doosan_e0509",
            asset_id="doosan_e0509",
        ),
        # NOTE: delta action transform 적용 여부
        # 수집 데이터가 absolute joint position이면 아래 활성화
        # use_delta_joint_actions=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    # Fine-tuning hyperparams (50 demos 기준)
    num_train_steps=5_000,    # 50 demos → 5k steps 충분
    batch_size=32,            # V100 16GB → bs=32 가능할 수도
    # LoRA 설정 (VRAM 절약)
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
        action_horizon=16,
        max_token_len=180,
    ).get_freeze_filter(),
    ema_decay=None,  # LoRA에서는 EMA off
),
"""


# ============================================================
# RepackTransform for Doosan E0509 (LeRobot format)
# ============================================================

"""
# openpi/src/openpi/transforms.py 또는 별도 파일에 추가

class DoosanE0509RepackTransform:
    '''데모 데이터의 key를 openpi가 기대하는 형식으로 매핑'''
    
    def __call__(self, sample):
        return {
            "images": {
                "cam_ext": sample["observation.images.cam_ext"],
                # wrist cam이 있다면:
                # "cam_wrist": sample["observation.images.cam_wrist"],
            },
            "state": sample["observation.state"],   # [j1..j6, grip] 7-dim
            "actions": sample["action"],              # [Δj1..Δj6, grip] 7-dim
        }
"""
```

---

## 4. Action Adapter (가장 중요 — 사전 준비 핵심)

### `doosan_action_adapter.py`

```python
"""
Doosan E0509 Action Adapter
VLA 출력 → 두산 로봇 제어 명령 변환

이 파일은 로봇 없이 미리 완성 가능.
현장에서는 SAFETY_CONFIG 숫자만 실측 확인.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DoosanSafetyConfig:
    """
    ⚠️ 현장 실측 후 업데이트할 값들
    """
    # Joint position limits (rad) — conservative
    joint_pos_lower: np.ndarray = None
    joint_pos_upper: np.ndarray = None
    
    # Max delta per timestep (rad) — 안전 제한
    max_delta_per_step: float = 0.05       # ~2.86° per step
    
    # Max velocity (rad/s)
    max_joint_velocity: float = 1.0        # ~57°/s (보수적)
    
    # Gripper
    gripper_threshold: float = 0.5         # > 0.5 → open, < 0.5 → close
    
    def __post_init__(self):
        if self.joint_pos_lower is None:
            # 보수적 기본값 (현장에서 반드시 실측 업데이트)
            self.joint_pos_lower = np.deg2rad([-350, -90, -140, -350, -350, -350])
            self.joint_pos_upper = np.deg2rad([+350, +90, +140, +350, +350, +350])


class DoosanActionAdapter:
    """
    VLA raw output → 두산 E0509 제어 명령 변환.
    
    Pipeline:
    1. Denormalize (norm stats 기반)
    2. Delta → absolute joint target 계산
    3. Safety clamp (position limits + velocity limits)
    4. Gripper 이진화
    5. 두산 servoj 형식으로 출력
    """
    
    def __init__(self, safety_config: DoosanSafetyConfig = None, norm_stats: dict = None):
        self.safety = safety_config or DoosanSafetyConfig()
        self.norm_stats = norm_stats  # {'action_mean': ..., 'action_std': ...}
        
        # State tracking
        self.current_joint_pos = None  # 로봇에서 읽어온 현재 joint position
        self.last_gripper_state = 0.0
        
        # Logging
        self.clamp_count = 0
        self.total_count = 0
    
    def set_current_state(self, joint_positions: np.ndarray, gripper: float):
        """
        로봇으로부터 현재 상태를 업데이트.
        매 timestep마다 호출.
        
        Args:
            joint_positions: [j1, j2, j3, j4, j5, j6] in radians
            gripper: 0.0 (closed) ~ 1.0 (open)
        """
        self.current_joint_pos = np.array(joint_positions, dtype=np.float64)
        self.last_gripper_state = gripper
    
    def convert(self, raw_action: np.ndarray, dt: float = 0.1) -> dict:
        """
        VLA raw output → 두산 제어 명령.
        
        Args:
            raw_action: VLA가 출력한 7-dim array [Δj1..Δj6, grip]
                       (normalized 상태일 수 있음)
            dt: timestep duration (seconds), default 0.1s = 10Hz
        
        Returns:
            dict with:
                'joint_targets': [j1..j6] absolute position targets (rad)
                'gripper_open': bool
                'was_clamped': bool (safety clamp 발동 여부)
                'clamp_ratio': float (이번까지의 clamp 비율)
        """
        assert self.current_joint_pos is not None, \
            "Call set_current_state() first!"
        assert raw_action.shape == (7,), \
            f"Expected 7-dim action, got {raw_action.shape}"
        
        self.total_count += 1
        was_clamped = False
        
        # --- Step 1: Denormalize ---
        action = self._denormalize(raw_action)
        
        # --- Step 2: Split joints & gripper ---
        delta_joints = action[:6]
        gripper_raw = action[6]
        
        # --- Step 3: Delta clamp (per-step limit) ---
        delta_joints, clamped_delta = self._clamp_delta(delta_joints)
        
        # --- Step 4: Compute absolute target ---
        target_joints = self.current_joint_pos + delta_joints
        
        # --- Step 5: Position clamp (joint limits) ---
        target_joints, clamped_pos = self._clamp_position(target_joints)
        
        # --- Step 6: Velocity clamp ---
        target_joints, clamped_vel = self._clamp_velocity(
            target_joints, self.current_joint_pos, dt
        )
        
        was_clamped = clamped_delta or clamped_pos or clamped_vel
        if was_clamped:
            self.clamp_count += 1
        
        # --- Step 7: Gripper ---
        gripper_open = gripper_raw > self.safety.gripper_threshold
        
        return {
            'joint_targets': target_joints,           # [6,] rad
            'gripper_open': bool(gripper_open),        # bool
            'was_clamped': was_clamped,
            'clamp_ratio': self.clamp_count / self.total_count,
        }
    
    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        """Undo normalization applied during training."""
        if self.norm_stats is not None:
            mean = self.norm_stats['action_mean']
            std = self.norm_stats['action_std']
            return action * std + mean
        return action.copy()
    
    def _clamp_delta(self, delta: np.ndarray):
        """Per-step delta magnitude clamp."""
        max_d = self.safety.max_delta_per_step
        clamped = np.any(np.abs(delta) > max_d)
        delta_clamped = np.clip(delta, -max_d, max_d)
        return delta_clamped, clamped
    
    def _clamp_position(self, target: np.ndarray):
        """Joint position limits clamp."""
        lo = self.safety.joint_pos_lower
        hi = self.safety.joint_pos_upper
        clamped = np.any(target < lo) or np.any(target > hi)
        target_clamped = np.clip(target, lo, hi)
        return target_clamped, clamped
    
    def _clamp_velocity(self, target: np.ndarray, current: np.ndarray, dt: float):
        """Velocity-based clamp: max change per dt."""
        max_change = self.safety.max_joint_velocity * dt
        diff = target - current
        clamped = np.any(np.abs(diff) > max_change)
        diff_clamped = np.clip(diff, -max_change, max_change)
        return current + diff_clamped, clamped


# ============================================================
# Unit Test (로봇 없이 실행 가능)
# ============================================================

def test_adapter():
    """로봇 없이 로직 검증"""
    adapter = DoosanActionAdapter()
    
    # 시뮬레이션: 현재 joint position = 모두 0
    adapter.set_current_state(
        joint_positions=np.zeros(6),
        gripper=0.0
    )
    
    # Case 1: 정상 범위 delta
    action = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8])
    result = adapter.convert(action)
    print(f"[Case 1] Normal delta:")
    print(f"  targets (deg): {np.rad2deg(result['joint_targets'])}")
    print(f"  gripper_open: {result['gripper_open']}")
    print(f"  was_clamped: {result['was_clamped']}")
    assert not result['was_clamped'], "Should not be clamped"
    assert result['gripper_open'], "Gripper should be open (0.8 > 0.5)"
    
    # Case 2: 과도한 delta → clamp 발동
    action_big = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
    result2 = adapter.convert(action_big)
    print(f"\n[Case 2] Excessive delta:")
    print(f"  targets (deg): {np.rad2deg(result2['joint_targets'])}")
    print(f"  was_clamped: {result2['was_clamped']}")
    assert result2['was_clamped'], "Should be clamped"
    assert not result2['gripper_open'], "Gripper should be closed (0.3 < 0.5)"
    
    # Case 3: Joint limit 근처에서 delta → position clamp
    adapter.set_current_state(
        joint_positions=np.deg2rad([349, 89, 0, 0, 0, 0]),
        gripper=1.0
    )
    action_edge = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.5])
    result3 = adapter.convert(action_edge)
    print(f"\n[Case 3] Near joint limit:")
    print(f"  targets (deg): {np.rad2deg(result3['joint_targets'])}")
    print(f"  was_clamped: {result3['was_clamped']}")
    
    print(f"\n✅ All tests passed! Clamp ratio: {result3['clamp_ratio']:.1%}")


if __name__ == "__main__":
    test_adapter()
```

---

## 5. ROS2 Deployment Node (사전 작성)

### `doosan_vla_controller.py`

```python
"""
ROS2 Node: VLA Inference Server → Doosan E0509 제어
로봇 없이 코드 구조만 미리 완성. 현장에서 IP/port만 설정.

실행:
  ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real model:=e0509 host:=192.168.127.100
  python doosan_vla_controller.py
"""

import time
import numpy as np
import requests
from typing import Optional

# ============================================================
# VLA Inference Client
# ============================================================

class VLAInferenceClient:
    """VLA inference server (port 8000)에 HTTP 요청"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
    
    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        """
        Args:
            image: RGB image (H, W, 3) uint8
            state: [j1..j6, grip] 7-dim
            instruction: 자연어 명령 (한국어)
        Returns:
            action: [Δj1..Δj6, grip] 7-dim (또는 action chunk)
        """
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Image → base64
        img_pil = Image.fromarray(image)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "image": img_b64,
            "state": state.tolist(),
            "instruction": instruction,
        }
        
        try:
            resp = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=2.0  # 2초 timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return np.array(data["actions"])  # shape: (horizon, 7) or (7,)
        except Exception as e:
            print(f"[VLA] Inference failed: {e}")
            return None


# ============================================================
# Doosan Robot Interface (via dsr_msgs2 / DRCF)
# ============================================================

class DoosanRobotInterface:
    """
    두산 로봇 ROS2 인터페이스.
    현장에서 실제 연결 후 테스트할 것.
    
    두 가지 방식 지원:
    1. ROS2 FollowJointTrajectory action
    2. 두산 DRCF TCP 직접 통신 (servoj)
    """
    
    def __init__(self, mode: str = "drcf", robot_ip: str = "192.168.127.100"):
        self.mode = mode
        self.robot_ip = robot_ip
        self.drcf_port = 12345
        
        if mode == "ros2":
            self._init_ros2()
        elif mode == "drcf":
            self._init_drcf()
    
    def _init_ros2(self):
        """ROS2 action client 초기화"""
        # 현장에서 구현
        # rclpy.init()
        # self.node = ...
        # self.action_client = ActionClient(node, FollowJointTrajectory, ...)
        pass
    
    def _init_drcf(self):
        """DRCF TCP 직접 연결"""
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 현장에서: self.sock.connect((self.robot_ip, self.drcf_port))
        print(f"[DRCF] Will connect to {self.robot_ip}:{self.drcf_port}")
    
    def get_current_state(self) -> dict:
        """
        현재 로봇 상태 읽기.
        Returns: {'joint_positions': [6,], 'gripper': float}
        """
        # --- 현장 구현 ---
        # ROS2: /dsr01e0509/joint_states topic subscribe
        # DRCF: get_current_posj() 호출
        
        # Placeholder (테스트용)
        return {
            'joint_positions': np.zeros(6),
            'gripper': 0.0,
        }
    
    def send_joint_target(self, joint_targets: np.ndarray, duration_sec: float = 0.1):
        """
        Joint position 명령 전송.
        
        Args:
            joint_targets: [j1..j6] in radians
            duration_sec: 이 시간 안에 도달
        """
        # 두산은 degree 단위로 받음!
        joint_deg = np.rad2deg(joint_targets).tolist()
        
        if self.mode == "drcf":
            # servoj(pos, vel, acc, time)
            # 현장에서: self._send_drcf_command(f"servoj({joint_deg}, 0, 0, {duration_sec})")
            pass
        elif self.mode == "ros2":
            # FollowJointTrajectory goal 전송
            pass
        
        # Debug
        print(f"[CMD] Target (deg): [{', '.join(f'{d:.1f}' for d in joint_deg)}]")
    
    def set_gripper(self, open_cmd: bool):
        """그리퍼 열기/닫기"""
        # 현장에서 구현: 두산 Tool I/O 또는 Robotiq 제어
        state_str = "OPEN" if open_cmd else "CLOSE"
        print(f"[GRIPPER] {state_str}")


# ============================================================
# Main Control Loop
# ============================================================

class VLAControlLoop:
    """
    메인 제어 루프: Camera → VLA → Adapter → Robot
    """
    
    def __init__(
        self,
        vla_url: str = "http://localhost:8000",
        robot_ip: str = "192.168.127.100",
        control_hz: float = 10.0,
        instruction: str = "저 긴 물체 좀 가져다줘",
    ):
        self.vla_client = VLAInferenceClient(vla_url)
        self.robot = DoosanRobotInterface(mode="drcf", robot_ip=robot_ip)
        
        from doosan_action_adapter import DoosanActionAdapter, DoosanSafetyConfig
        self.adapter = DoosanActionAdapter(
            safety_config=DoosanSafetyConfig(),
            norm_stats=None,  # 학습 후 채워넣기
        )
        
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.instruction = instruction
        self.camera = None  # 현장에서 카메라 객체 연결
    
    def get_camera_image(self) -> np.ndarray:
        """카메라에서 RGB 이미지 캡처"""
        # 현장 구현: RealSense / USB cam / ROS2 topic
        # Placeholder
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def run(self, max_steps: int = 200):
        """
        메인 루프 실행.
        max_steps=200 at 10Hz = 20초 에피소드
        """
        print(f"=== VLA Control Loop Start ===")
        print(f"  Instruction: {self.instruction}")
        print(f"  Control Hz: {self.control_hz}")
        print(f"  Max steps: {max_steps}")
        print()
        
        for step in range(max_steps):
            t0 = time.time()
            
            # 1. 로봇 상태 읽기
            state = self.robot.get_current_state()
            joint_pos = state['joint_positions']
            gripper = state['gripper']
            
            # Adapter에 현재 상태 전달
            obs_state = np.concatenate([joint_pos, [gripper]])  # [7,]
            self.adapter.set_current_state(joint_pos, gripper)
            
            # 2. 카메라 이미지
            image = self.get_camera_image()
            
            # 3. VLA inference
            raw_action = self.vla_client.predict(image, obs_state, self.instruction)
            
            if raw_action is None:
                print(f"[Step {step}] VLA inference failed, holding position")
                continue
            
            # action chunk인 경우 첫 번째만 사용
            if raw_action.ndim == 2:
                raw_action = raw_action[0]
            
            # 4. Action adapter (safety clamp 포함)
            cmd = self.adapter.convert(raw_action, dt=self.dt)
            
            # 5. 로봇에 명령 전송
            self.robot.send_joint_target(cmd['joint_targets'], duration_sec=self.dt)
            self.robot.set_gripper(cmd['gripper_open'])
            
            # 6. Timing
            elapsed = time.time() - t0
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[Step {step}] ⚠️ Overrun: {elapsed:.3f}s > {self.dt:.3f}s")
            
            # 7. Logging
            if step % 20 == 0:
                print(f"[Step {step:3d}] clamp_ratio={cmd['clamp_ratio']:.1%} "
                      f"elapsed={elapsed*1000:.0f}ms")
        
        print(f"\n=== Control Loop Done ===")
        print(f"  Total clamp ratio: {cmd['clamp_ratio']:.1%}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-url", default="http://localhost:8000")
    parser.add_argument("--robot-ip", default="192.168.127.100")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--instruction", default="저 긴 물체 좀 가져다줘")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    
    loop = VLAControlLoop(
        vla_url=args.vla_url,
        robot_ip=args.robot_ip,
        control_hz=args.hz,
        instruction=args.instruction,
    )
    loop.run(max_steps=args.max_steps)
```

---

## 6. Demo Data Recorder (사전 작성)

### `demo_recorder.py`

```python
"""
데모 수집 스크립트.
두산 Direct Teaching (Hand Guiding) + 카메라 동기 녹화.

현장에서 실행:
  python demo_recorder.py --robot-ip 192.168.127.100 --save-dir ./demos
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class DemoRecorder:
    """
    Direct Teaching 데모 녹화기.
    두산 Cockpit 버튼으로 teaching mode 진입 후, 
    이 스크립트가 joint positions + camera를 동기 녹화.
    """
    
    def __init__(self, save_dir: str, robot_ip: str, record_hz: float = 10.0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.robot_ip = robot_ip
        self.record_hz = record_hz
        self.dt = 1.0 / record_hz
        
        # 현장에서 초기화
        self.robot = None  # DoosanRobotInterface
        self.camera = None  # cv2.VideoCapture or RealSense
    
    def record_episode(self, task_instruction: str, episode_id: int) -> dict:
        """
        1 에피소드 녹화.
        Direct Teaching 중 joint states + camera를 동기 기록.
        
        Returns:
            episode dict with 'states', 'actions', 'images', 'instruction'
        """
        print(f"\n{'='*50}")
        print(f"  Episode {episode_id}")
        print(f"  Instruction: {task_instruction}")
        print(f"  [Enter]를 눌러 녹화 시작, [q]를 눌러 종료")
        print(f"{'='*50}")
        input("Press Enter to START recording...")
        
        states = []
        images = []
        timestamps = []
        
        print("Recording... (press Ctrl+C to stop)")
        
        try:
            while True:
                t0 = time.time()
                
                # 로봇 상태 읽기
                state = self._get_robot_state()  # [j1..j6, grip] 7-dim
                states.append(state)
                
                # 카메라 캡처
                img = self._get_camera_image()  # (H, W, 3) uint8
                images.append(img)
                
                timestamps.append(time.time())
                
                # Timing
                elapsed = time.time() - t0
                if self.dt - elapsed > 0:
                    time.sleep(self.dt - elapsed)
                    
        except KeyboardInterrupt:
            print(f"\nRecording stopped. {len(states)} frames captured.")
        
        # Compute actions (delta between consecutive states)
        states_arr = np.array(states)
        actions = np.diff(states_arr, axis=0)  # (T-1, 7) delta
        # 마지막 action은 0 (정지)
        actions = np.vstack([actions, np.zeros((1, 7))])
        
        episode = {
            'instruction': task_instruction,
            'states': states_arr,           # (T, 7)
            'actions': actions,             # (T, 7) delta
            'images': np.array(images),     # (T, H, W, 3)
            'timestamps': timestamps,
            'metadata': {
                'episode_id': episode_id,
                'record_hz': self.record_hz,
                'robot': 'doosan_e0509',
                'recorded_at': datetime.now().isoformat(),
            }
        }
        
        # Save
        ep_dir = self.save_dir / f"episode_{episode_id:04d}"
        ep_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            ep_dir / "data.npz",
            states=episode['states'],
            actions=episode['actions'],
        )
        # Save images separately (큰 파일)
        np.savez_compressed(ep_dir / "images.npz", images=episode['images'])
        # Save metadata
        with open(ep_dir / "metadata.json", 'w') as f:
            json.dump(episode['metadata'], f, indent=2)
            json.dump({'instruction': task_instruction}, f)
        
        print(f"Saved to {ep_dir}")
        return episode
    
    def _get_robot_state(self) -> np.ndarray:
        """현장 구현: 로봇 joint state + gripper 읽기"""
        # Placeholder
        return np.zeros(7)
    
    def _get_camera_image(self) -> np.ndarray:
        """현장 구현: 카메라 캡처"""
        # Placeholder
        return np.zeros((224, 224, 3), dtype=np.uint8)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="./demos")
    parser.add_argument("--robot-ip", default="192.168.127.100")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--instruction", default="파란 약병을 트레이에 옮겨줘")
    parser.add_argument("--num-episodes", type=int, default=50)
    args = parser.parse_args()
    
    recorder = DemoRecorder(args.save_dir, args.robot_ip, args.hz)
    
    for i in range(args.num_episodes):
        recorder.record_episode(args.instruction, episode_id=i)
        
        if i < args.num_episodes - 1:
            cont = input(f"\n다음 에피소드 ({i+1}/{args.num_episodes})? [Enter/q]: ")
            if cont.lower() == 'q':
                break
    
    print(f"\n✅ Done! {i+1} episodes recorded in {args.save_dir}")
```

---

## 7. 현장 체크리스트

### 도착 즉시 (첫 30분)
- [ ] 두산 E0509 모델 번호 확인 (E0509인지 다른 모델인지)
- [ ] 컨트롤러 IP 확인 (`192.168.127.100`이 맞는지)
- [ ] 네트워크 연결 확인 (노트북 → 컨트롤러 ping)
- [ ] Teach pendant에서 joint limits 설정 확인 → config 업데이트
- [ ] **Joint 2 실제 범위 확인** (±95° vs ±360° — 설치 방식에 따라 다름)
- [ ] Gripper 종류 확인 (Robotiq 2F? Pneumatic? 두산 내장?)
- [ ] 카메라 마운트 위치 결정 + 고정

### Config 업데이트 (15분)
- [ ] `doosan_e0509_config.py` → joint limits 실측값으로 교체
- [ ] `DoosanSafetyConfig` → joint_pos_lower/upper 업데이트
- [ ] Gripper 제어 방식에 맞게 `set_gripper()` 구현
- [ ] 카메라 해상도/FPS 확인 → `get_camera_image()` 구현

### 통신 테스트 (30분)
- [ ] `ros2 launch dsr_bringup2 ... mode:=real model:=e0509` 실행
- [ ] `/dsr01e0509/joint_states` topic 수신 확인
- [ ] servoj 명령으로 1° 움직여보기
- [ ] Emergency stop 작동 확인

---

## 8. 미리 준비 가능 / 불가능 정리

| 항목 | 미리 준비 | 현장 필요 |
|------|----------|----------|
| Embodiment config (joint names, dims) | ✅ 완료 | 숫자만 실측 확인 |
| Action adapter + safety clamp | ✅ 완료 | limits 숫자 업데이트 |
| ROS2 controller 코드 구조 | ✅ 완료 | IP/port + 통신 테스트 |
| Demo recorder 코드 | ✅ 완료 | 카메라/로봇 연결 |
| VLA inference client | ✅ 완료 | VLA 서버 URL만 변경 |
| GR00T/OpenPI train config | ✅ 완료 | norm stats 계산 |
| Norm stats | ❌ | 데모 수집 후 계산 |
| 카메라 캘리브레이션 | ❌ | 현장 카메라로 |
| 실제 데모 수집 | ❌ | 실물 로봇 필요 |
| Fine-tuning | ❌ | 데모 + GPU 필요 |
| End-to-end 테스트 | ❌ | 모든 것 연결 후 |
