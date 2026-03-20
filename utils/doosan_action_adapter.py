"""
VLA output → 두산 E0509 로봇 명령 변환 (3중 safety clamp)
로봇 없이 미리 완성 + 테스트 가능. 현장에서는 config 숫자만 업데이트.
"""

import os, sys
import numpy as np
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *


@dataclass
class DoosanSafetyConfig:
    joint_pos_lower: np.ndarray = field(default=None)
    joint_pos_upper: np.ndarray = field(default=None)
    max_delta_per_step: float = MAX_DELTA_PER_STEP
    max_joint_velocity: float = MAX_JOINT_VELOCITY
    gripper_threshold: float = GRIPPER_THRESHOLD

    def __post_init__(self):
        if self.joint_pos_lower is None:
            self.joint_pos_lower = JOINT_LIMITS_LOWER.copy()
            self.joint_pos_upper = JOINT_LIMITS_UPPER.copy()


class DoosanActionAdapter:
    def __init__(self, safety_config=None, norm_stats=None):
        self.safety = safety_config or DoosanSafetyConfig()
        self.norm_stats = norm_stats
        self.current_joint_pos = None
        self.last_gripper = 0.0
        self.clamp_count = 0
        self.total_count = 0

    def set_current_state(self, joint_positions, gripper):
        self.current_joint_pos = np.array(joint_positions, dtype=np.float64)
        self.last_gripper = gripper

    def convert(self, raw_action, dt=0.1):
        assert self.current_joint_pos is not None, "Call set_current_state() first"
        assert raw_action.shape == (ACTION_DIM,), f"Expected {ACTION_DIM}-dim, got {raw_action.shape}"
        self.total_count += 1

        action = self._denormalize(raw_action)
        delta, c1 = self._clamp_delta(action[:NUM_JOINTS])
        target = self.current_joint_pos + delta
        target, c2 = self._clamp_position(target)
        target, c3 = self._clamp_velocity(target, self.current_joint_pos, dt)

        clamped = c1 or c2 or c3
        if clamped:
            self.clamp_count += 1

        return {
            "joint_targets": target,
            "gripper_open": bool(action[NUM_JOINTS] > self.safety.gripper_threshold),
            "was_clamped": clamped,
            "clamp_ratio": self.clamp_count / self.total_count,
        }

    def _denormalize(self, action):
        if self.norm_stats:
            return action * self.norm_stats["action_std"] + self.norm_stats["action_mean"]
        return action.copy()

    def _clamp_delta(self, delta):
        m = self.safety.max_delta_per_step
        c = bool(np.any(np.abs(delta) > m))
        return np.clip(delta, -m, m), c

    def _clamp_position(self, target):
        lo, hi = self.safety.joint_pos_lower, self.safety.joint_pos_upper
        c = bool(np.any(target < lo) or np.any(target > hi))
        return np.clip(target, lo, hi), c

    def _clamp_velocity(self, target, current, dt):
        mc = self.safety.max_joint_velocity * dt
        diff = target - current
        c = bool(np.any(np.abs(diff) > mc))
        return current + np.clip(diff, -mc, mc), c


if __name__ == "__main__":
    adapter = DoosanActionAdapter()
    adapter.set_current_state(np.zeros(NUM_JOINTS), 0.0)
    r = adapter.convert(np.array([0.01]*NUM_JOINTS + [0.8]))
    print(f"Normal: clamped={r['was_clamped']}, grip={r['gripper_open']}")
    r = adapter.convert(np.array([0.5]*NUM_JOINTS + [0.3]))
    print(f"Excessive: clamped={r['was_clamped']}, grip={r['gripper_open']}")
    print("✅ Adapter OK")
