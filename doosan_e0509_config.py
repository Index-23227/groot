"""
Doosan E0509 Embodiment Configuration
GR00T N1.6 / OpenPI (π₀.5) 공용 config

미리 작성 가능 — 현장에서 joint limits만 실측 확인 후 업데이트
"""

import numpy as np


# ============================================================
# Doosan E0509 Embodiment Config
# ============================================================

DOOSAN_E0509_CONFIG = {
    "embodiment_name": "doosan_e0509",
    "num_joints": 6,
    "num_actions": 7,  # 6 joints + 1 gripper
    
    # Joint Names (ROS2 doosan-robot2 convention)
    "joint_names": [
        "joint1", "joint2", "joint3",
        "joint4", "joint5", "joint6",
    ],
    
    # Joint Limits (radians)
    # ⚠️ 현장 실측 후 업데이트할 것
    # J2: 바닥 설치 시 ±95° 권장 (자기충돌 방지)
    "joint_limits_lower": np.deg2rad([-350, -95, -145, -350, -350, -350]),
    "joint_limits_upper": np.deg2rad([+350, +95, +145, +350, +350, +350]),
    
    # Joint Max Velocities (rad/s)
    "joint_max_velocities": np.deg2rad([120, 120, 150, 225, 225, 225]),
    
    # Gripper
    "gripper_range": [0.0, 1.0],  # 0=closed, 1=open
    
    # Action Space
    "action_type": "joint_delta",   # state-relative delta
    "action_dim": 7,                # [Δj1..Δj6, grip]
    
    # Observation Space
    "state_dim": 7,                 # [j1..j6, grip]
    
    # Control
    "control_frequency_hz": 10,
    "action_horizon": 16,           # action chunk size
    
    # Robot Physical Specs
    "payload_kg": 5.0,
    "reach_mm": 900,
    "repeatability_mm": 0.05,
    "dof": 6,
    "protection_rating": "IP66",
    
    # Network (defaults)
    "controller_ip": "192.168.127.100",
    "controller_port": 12345,
}


# ============================================================
# Normalization Statistics
# ============================================================

class NormStats:
    """데모 수집 후 계산하는 normalization statistics"""
    
    def __init__(self):
        # Placeholder — 데모 수집 후 compute()로 채움
        self.action_mean = np.zeros(7)
        self.action_std = np.ones(7)
        self.state_mean = np.zeros(7)
        self.state_std = np.ones(7)
        self.computed = False
    
    def compute(self, demo_episodes: list):
        """
        데모 수집 후 호출.
        
        Args:
            demo_episodes: list of dicts, each with 'actions' (T,7) and 'states' (T,7)
        """
        all_actions = np.concatenate([ep['actions'] for ep in demo_episodes], axis=0)
        all_states = np.concatenate([ep['states'] for ep in demo_episodes], axis=0)
        
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0).clip(min=1e-6)
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0).clip(min=1e-6)
        self.computed = True
        
        print(f"Norm stats computed from {len(demo_episodes)} episodes, "
              f"{len(all_actions)} frames")
        print(f"  action_mean: {self.action_mean.round(4)}")
        print(f"  action_std:  {self.action_std.round(4)}")
    
    def to_dict(self):
        return {
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }
    
    def save(self, path: str):
        np.savez(path, **self.to_dict())
        print(f"Saved norm stats to {path}")
    
    def load(self, path: str):
        data = np.load(path)
        self.action_mean = data['action_mean']
        self.action_std = data['action_std']
        self.state_mean = data['state_mean']
        self.state_std = data['state_std']
        self.computed = True
        print(f"Loaded norm stats from {path}")


# ============================================================
# OpenPI TrainConfig Template
# ============================================================

OPENPI_CONFIG_TEMPLATE = """
# --- openpi/src/openpi/training/config.py에 추가 ---

TrainConfig(
    name="pi05_doosan_e0509",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,       # 6 joints + 1 gripper
        action_horizon=16,
        max_token_len=180,
    ),
    data=LeRobotDataConfig(
        repo_id="local/doosan_e0509_demos",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="./assets/doosan_e0509",
            asset_id="doosan_e0509",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=5_000,
    batch_size=32,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
        action_horizon=16,
        max_token_len=180,
    ).get_freeze_filter(),
    ema_decay=None,
),
"""


def get_config():
    return DOOSAN_E0509_CONFIG


def print_config():
    config = DOOSAN_E0509_CONFIG
    print(f"=== Doosan E0509 Embodiment Config ===")
    print(f"  Joints: {config['num_joints']}")
    print(f"  Action dim: {config['action_dim']} ({config['action_type']})")
    print(f"  Control Hz: {config['control_frequency_hz']}")
    print(f"  Joint limits (deg):")
    lo = np.rad2deg(config['joint_limits_lower'])
    hi = np.rad2deg(config['joint_limits_upper'])
    for i, name in enumerate(config['joint_names']):
        print(f"    {name}: [{lo[i]:.0f}°, {hi[i]:.0f}°]")
    print(f"\n  OpenPI config template:")
    print(OPENPI_CONFIG_TEMPLATE)


if __name__ == "__main__":
    print_config()
