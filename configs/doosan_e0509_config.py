"""
Doosan E0509 — 중앙 설정 파일
모든 스크립트가 이 파일을 import해서 로봇 스펙을 참조.

⚠️ 현장 도착 후 반드시 업데이트:
  CONTROLLER_IP, JOINT_LIMITS, CAMERA 설정
"""

import numpy as np

# =============================================================
# 🔧 현장에서 업데이트할 항목
# =============================================================

CONTROLLER_IP = "110.120.1.39"
CONTROLLER_PORT = 12345
ROBOT_MODEL = "e0509"

CAMERA_TYPE = "realsense"  # "opencv" / "realsense" / "ros2_topic"
CAMERA_INDEX = 0
CAMERA_TOPIC = "/camera/color/image_raw"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# =============================================================
# 로봇 스펙
# =============================================================

NUM_JOINTS = 6
GRIPPER_ACTION_DIM = 1
ACTION_DIM = NUM_JOINTS + GRIPPER_ACTION_DIM  # 7

JOINT_LIMITS_LOWER = np.deg2rad([-350, -95, -145, -350, -350, -350])
JOINT_LIMITS_UPPER = np.deg2rad([+350, +95, +145, +350, +350, +350])
JOINT_MAX_VELOCITIES = np.deg2rad([120, 120, 150, 225, 225, 225])

# Joint 이름: 실제 /joint_states 토픽에서 발행되는 이름과 일치해야 함
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# =============================================================
# 그리퍼: ROBOTIS RH-P12-RN-A
# =============================================================

GRIPPER_TYPE = "robotis_rh_p12_rn_a"

# 그리퍼 stroke 범위 (하드웨어 단위)
GRIPPER_STROKE_MIN = 0      # 완전 열림
GRIPPER_STROKE_MAX = 700    # 완전 닫힘

# /joint_states에 포함되는 그리퍼 joint 이름 (4개이지만 1-DOF, 전부 동일 값)
GRIPPER_JOINT_NAMES = ["gripper_rh_r1", "gripper_rh_r2", "gripper_rh_l1", "gripper_rh_l2"]

# ROS2 인터페이스
JOINT_STATE_TOPIC = "/joint_states"                        # arm 6 + gripper 4 (10개)
GRIPPER_OPEN_SERVICE = "/dsr01/gripper/open"               # std_srvs/Trigger
GRIPPER_CLOSE_SERVICE = "/dsr01/gripper/close"             # std_srvs/Trigger
GRIPPER_POSITION_TOPIC = "/dsr01/gripper/position_cmd"     # std_msgs/Int32 (0~700)
GRIPPER_STROKE_TOPIC = "/dsr01/gripper/stroke"             # std_msgs/Int32 (현재 stroke)
MOVE_JOINT_SERVICE = "/dsr01/motion/move_joint"            # dsr_msgs2/MoveJoint (degree!)
MOVE_LINE_SERVICE  = "/dsr01/motion/move_line"             # dsr_msgs2/MoveLine — Cartesian (mm + deg)
GET_TCP_SERVICE    = "/dsr01/aux_control/get_current_posx" # dsr_msgs2/GetCurrentPosx → pos[6]

# ⚠️ 두산은 degree 단위, VLA는 radian 단위
# 이 변환은 반드시 adapter 한 곳에서만 수행
JOINT_UNIT = "degree"

# =============================================================
# 그리퍼 ↔ VLA 변환 함수
# =============================================================
# VLA action space에서 grip 값:
#   0.0 = 열림 (stroke 0)
#   1.0 = 닫힘 (stroke 700)

def grip_to_stroke(grip_value: float) -> int:
    """VLA grip 값 (0.0~1.0) → 그리퍼 stroke (0~700)"""
    return int(np.clip(grip_value, 0.0, 1.0) * GRIPPER_STROKE_MAX)

def stroke_to_grip(stroke: int) -> float:
    """그리퍼 stroke (0~700) → VLA grip 값 (0.0~1.0)"""
    return float(np.clip(stroke, GRIPPER_STROKE_MIN, GRIPPER_STROKE_MAX)) / GRIPPER_STROKE_MAX

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
GRIPPER_THRESHOLD = 0.5     # grip > 0.5 → 닫힘, grip ≤ 0.5 → 열림

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
    print(f"  Gripper: {GRIPPER_TYPE} (stroke {GRIPPER_STROKE_MIN}~{GRIPPER_STROKE_MAX})")
    print(f"  Action dim: {ACTION_DIM} (joints={NUM_JOINTS} + grip={GRIPPER_ACTION_DIM})")
    print(f"  Control Hz: {CONTROL_HZ}, Action horizon: {ACTION_HORIZON}")
    print(f"  Camera: {CAMERA_TYPE} (index={CAMERA_INDEX})")
    print(f"  Safety: max_delta={np.rad2deg(MAX_DELTA_PER_STEP):.1f}°/step")
    print(f"  ROS2 Topics:")
    print(f"    joint_states: {JOINT_STATE_TOPIC}")
    print(f"    gripper cmd:  {GRIPPER_POSITION_TOPIC}")
    print(f"    gripper fb:   {GRIPPER_STROKE_TOPIC}")
    print(f"  Joint limits (deg):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"    {name}: [{np.rad2deg(JOINT_LIMITS_LOWER[i]):.0f}°, {np.rad2deg(JOINT_LIMITS_UPPER[i]):.0f}°]")
    print(f"  Grip convention: 0.0=open(stroke 0), 1.0=close(stroke 700)")
    print(f"  ⚠️  두산 joint 단위: degree / VLA 단위: radian")
    print(f"  GR00T: {GROOT_NUM_GPUS} GPUs, {GROOT_MAX_STEPS} steps")
    print(f"  SmolVLA: GPU {SMOLVLA_GPU_ID}, {SMOLVLA_MAX_STEPS} steps")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
