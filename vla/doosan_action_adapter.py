"""
VLA output → 두산 E0509 로봇 명령 변환 (3중 safety clamp)
로봇 없이 미리 완성 + 테스트 가능. 현장에서는 config 숫자만 업데이트.

그리퍼 convention:
  VLA action[6]: 0.0 = 열림, 1.0 = 닫힘
  그리퍼 stroke: 0 = 열림, 700 = 닫힘
  grip > GRIPPER_THRESHOLD(0.5) → 닫힘 (close)

⚠️ 두산은 degree 단위, VLA는 radian 단위.
   이 변환은 이 adapter에서만 수행. 다른 곳에서 변환하지 말 것.
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
    """VLA action (radian) → 두산 로봇 명령 (degree + stroke) 변환.

    입력: raw_action [Δj1..Δj6, grip] (7-dim, radian)
    출력: {
        "joint_targets_rad": target joints (radian, safety clamped),
        "joint_targets_deg": target joints (degree, 두산 servoj용),
        "gripper_close": True면 닫기, False면 열기,
        "gripper_stroke": 0~700 (연속 제어용),
        ...
    }
    """

    def __init__(self, safety_config=None, norm_stats=None):
        self.safety = safety_config or DoosanSafetyConfig()
        self.norm_stats = norm_stats
        self.current_joint_pos = None
        self.last_gripper = 0.0
        self.clamp_count = 0
        self.total_count = 0

    def set_current_state(self, joint_positions, gripper):
        """현재 로봇 상태 설정.

        Args:
            joint_positions: 6-dim (radian)
            gripper: 0.0~1.0 (0=열림, 1=닫힘). stroke_to_grip()로 변환한 값.
        """
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

        grip_value = float(action[NUM_JOINTS])
        gripper_close = bool(grip_value > self.safety.gripper_threshold)

        return {
            "joint_targets_rad": target,
            "joint_targets_deg": np.rad2deg(target),
            "gripper_close": gripper_close,
            "gripper_open": not gripper_close,  # 하위 호환
            "gripper_stroke": grip_to_stroke(grip_value),
            "grip_value": grip_value,
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

    # grip=0.8 → 닫힘 (close)
    r = adapter.convert(np.array([0.01]*NUM_JOINTS + [0.8]))
    print(f"grip=0.8: close={r['gripper_close']}, stroke={r['gripper_stroke']}, clamped={r['was_clamped']}")
    assert r['gripper_close'] == True, "grip 0.8 should be CLOSE"
    assert r['gripper_stroke'] == 560, f"stroke should be 560, got {r['gripper_stroke']}"

    # grip=0.2 → 열림 (open)
    r = adapter.convert(np.array([0.01]*NUM_JOINTS + [0.2]))
    print(f"grip=0.2: close={r['gripper_close']}, stroke={r['gripper_stroke']}, clamped={r['was_clamped']}")
    assert r['gripper_close'] == False, "grip 0.2 should be OPEN"
    assert r['gripper_stroke'] == 140, f"stroke should be 140, got {r['gripper_stroke']}"

    # deg 변환 확인
    adapter.set_current_state(np.deg2rad([10, 20, 30, 40, 50, 60]), 0.0)
    r = adapter.convert(np.array([0.01]*NUM_JOINTS + [0.5]))
    print(f"deg: {r['joint_targets_deg']}")

    # 과도한 delta → clamp
    r = adapter.convert(np.array([0.5]*NUM_JOINTS + [0.3]))
    print(f"Excessive: clamped={r['was_clamped']}")

    print("✅ Adapter OK")
