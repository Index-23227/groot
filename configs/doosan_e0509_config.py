"""
Doosan E0509 — 중앙 설정 파일
모든 스크립트가 이 파일을 import해서 로봇 스펙을 참조.

⚠️ 현장 도착 후 반드시 업데이트:
  CONTROLLER_IP, JOINT_LIMITS, GRIPPER_TYPE, CAMERA 설정
"""

import numpy as np

# =============================================================
# 🔧 현장에서 업데이트할 항목
# =============================================================

CONTROLLER_IP = "192.168.127.100"
CONTROLLER_PORT = 12345
ROBOT_MODEL = "e0509"

GRIPPER_TYPE = "unknown"   # "robotiq_2f" / "pneumatic" / "doosan_builtin"
GRIPPER_ACTION_DIM = 1

CAMERA_TYPE = "opencv"     # "opencv" / "realsense" / "ros2_topic"
CAMERA_INDEX = 0
CAMERA_TOPIC = "/camera/color/image_raw"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# =============================================================
# 로봇 스펙 (현장 실측 후 보정)
# =============================================================

NUM_JOINTS = 6
ACTION_DIM = NUM_JOINTS + GRIPPER_ACTION_DIM  # 7

JOINT_LIMITS_LOWER = np.deg2rad([-350, -95, -145, -350, -350, -350])
JOINT_LIMITS_UPPER = np.deg2rad([+350, +95, +145, +350, +350, +350])
JOINT_MAX_VELOCITIES = np.deg2rad([120, 120, 150, 225, 225, 225])
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# =============================================================
# 제어 파라미터
# =============================================================

CONTROL_HZ = 10
ACTION_HORIZON = 8    # GR00T default=16, 8 권장 (error accumulation 줄이기)
DENOISING_STEPS = 4

# =============================================================
# Safety Clamp
# =============================================================

MAX_DELTA_PER_STEP = 0.05   # rad (~2.86°)
MAX_JOINT_VELOCITY = 1.0    # rad/s (~57°/s)
GRIPPER_THRESHOLD = 0.5

# =============================================================
# 학습 파라미터
# =============================================================

GROOT_BASE_MODEL = "nvidia/GR00T-N1.6-3B"
GROOT_MAX_STEPS = 10000
GROOT_BATCH_SIZE = 64
GROOT_SAVE_STEPS = 2000
GROOT_NUM_GPUS = 4
GROOT_GPU_IDS = "0,1,2,3"

SMOLVLA_BASE_MODEL = "lerobot/smolvla_base"
SMOLVLA_MAX_STEPS = 20000
SMOLVLA_BATCH_SIZE = 64
SMOLVLA_GPU_ID = "4"

# =============================================================
# 경로
# =============================================================

RAW_DATA_DIR = "./data/raw"
LEROBOT_DATA_DIR = "./data/lerobot_dataset"
GROOT_CHECKPOINT_DIR = "./checkpoints/groot"
SMOLVLA_CHECKPOINT_DIR = "./checkpoints/smolvla"

# =============================================================
# Utility
# =============================================================

def get_safety_config():
    return {
        "joint_pos_lower": JOINT_LIMITS_LOWER,
        "joint_pos_upper": JOINT_LIMITS_UPPER,
        "max_delta_per_step": MAX_DELTA_PER_STEP,
        "max_joint_velocity": MAX_JOINT_VELOCITY,
        "gripper_threshold": GRIPPER_THRESHOLD,
    }

def print_config():
    print("=" * 50)
    print("  Doosan E0509 Configuration")
    print("=" * 50)
    print(f"  Controller: {CONTROLLER_IP}:{CONTROLLER_PORT}")
    print(f"  Model: {ROBOT_MODEL}")
    print(f"  Gripper: {GRIPPER_TYPE}")
    print(f"  Action dim: {ACTION_DIM} (joints={NUM_JOINTS} + grip={GRIPPER_ACTION_DIM})")
    print(f"  Control Hz: {CONTROL_HZ}, Action horizon: {ACTION_HORIZON}")
    print(f"  Camera: {CAMERA_TYPE} (index={CAMERA_INDEX})")
    print(f"  Safety: max_delta={np.rad2deg(MAX_DELTA_PER_STEP):.1f}°/step")
    print(f"  Joint limits (deg):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"    {name}: [{np.rad2deg(JOINT_LIMITS_LOWER[i]):.0f}°, {np.rad2deg(JOINT_LIMITS_UPPER[i]):.0f}°]")
    print(f"  GR00T: {GROOT_NUM_GPUS} GPUs, {GROOT_MAX_STEPS} steps")
    print(f"  SmolVLA: GPU {SMOLVLA_GPU_ID}, {SMOLVLA_MAX_STEPS} steps")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
