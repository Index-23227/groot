"""
Doosan E0509 Action Adapter
VLA 출력 → 두산 로봇 제어 명령 변환

로봇 없이 미리 완성 가능.
현장에서는 DoosanSafetyConfig 숫자만 실측 확인.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DoosanSafetyConfig:
    """
    ⚠️ 현장 실측 후 업데이트할 값들
    """
    # Joint position limits (rad) — conservative
    joint_pos_lower: np.ndarray = field(default=None)
    joint_pos_upper: np.ndarray = field(default=None)
    
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
        self.current_joint_pos = None
        self.last_gripper_state = 0.0
        
        # Logging
        self.clamp_count = 0
        self.total_count = 0
    
    def set_current_state(self, joint_positions: np.ndarray, gripper: float):
        """
        로봇으로부터 현재 상태를 업데이트.
        매 timestep마다 호출.
        """
        self.current_joint_pos = np.array(joint_positions, dtype=np.float64)
        self.last_gripper_state = gripper
    
    def convert(self, raw_action: np.ndarray, dt: float = 0.1) -> dict:
        """
        VLA raw output → 두산 제어 명령.
        
        Args:
            raw_action: VLA가 출력한 7-dim array [Δj1..Δj6, grip]
            dt: timestep duration (seconds)
        
        Returns:
            dict with joint_targets, gripper_open, was_clamped, clamp_ratio
        """
        assert self.current_joint_pos is not None, "Call set_current_state() first!"
        assert raw_action.shape == (7,), f"Expected 7-dim action, got {raw_action.shape}"
        
        self.total_count += 1
        was_clamped = False
        
        # Step 1: Denormalize
        action = self._denormalize(raw_action)
        
        # Step 2: Split joints & gripper
        delta_joints = action[:6]
        gripper_raw = action[6]
        
        # Step 3: Delta clamp (per-step limit)
        delta_joints, clamped_delta = self._clamp_delta(delta_joints)
        
        # Step 4: Compute absolute target
        target_joints = self.current_joint_pos + delta_joints
        
        # Step 5: Position clamp (joint limits)
        target_joints, clamped_pos = self._clamp_position(target_joints)
        
        # Step 6: Velocity clamp
        target_joints, clamped_vel = self._clamp_velocity(
            target_joints, self.current_joint_pos, dt
        )
        
        was_clamped = clamped_delta or clamped_pos or clamped_vel
        if was_clamped:
            self.clamp_count += 1
        
        # Step 7: Gripper
        gripper_open = gripper_raw > self.safety.gripper_threshold
        
        return {
            'joint_targets': target_joints,
            'gripper_open': bool(gripper_open),
            'was_clamped': was_clamped,
            'clamp_ratio': self.clamp_count / self.total_count,
        }
    
    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        if self.norm_stats is not None:
            mean = self.norm_stats['action_mean']
            std = self.norm_stats['action_std']
            return action * std + mean
        return action.copy()
    
    def _clamp_delta(self, delta: np.ndarray):
        max_d = self.safety.max_delta_per_step
        clamped = bool(np.any(np.abs(delta) > max_d))
        delta_clamped = np.clip(delta, -max_d, max_d)
        return delta_clamped, clamped
    
    def _clamp_position(self, target: np.ndarray):
        lo = self.safety.joint_pos_lower
        hi = self.safety.joint_pos_upper
        clamped = bool(np.any(target < lo) or np.any(target > hi))
        target_clamped = np.clip(target, lo, hi)
        return target_clamped, clamped
    
    def _clamp_velocity(self, target: np.ndarray, current: np.ndarray, dt: float):
        max_change = self.safety.max_joint_velocity * dt
        diff = target - current
        clamped = bool(np.any(np.abs(diff) > max_change))
        diff_clamped = np.clip(diff, -max_change, max_change)
        return current + diff_clamped, clamped


def test_adapter():
    """로봇 없이 로직 검증"""
    adapter = DoosanActionAdapter()
    
    # 현재 joint position = 모두 0
    adapter.set_current_state(joint_positions=np.zeros(6), gripper=0.0)
    
    # Case 1: 정상 범위 delta
    action = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8])
    result = adapter.convert(action)
    print(f"[Case 1] Normal delta:")
    print(f"  targets (deg): {np.rad2deg(result['joint_targets']).round(2)}")
    print(f"  gripper_open: {result['gripper_open']}")
    print(f"  was_clamped: {result['was_clamped']}")
    assert not result['was_clamped']
    assert result['gripper_open']
    
    # Case 2: 과도한 delta → clamp
    action_big = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3])
    result2 = adapter.convert(action_big)
    print(f"\n[Case 2] Excessive delta:")
    print(f"  targets (deg): {np.rad2deg(result2['joint_targets']).round(2)}")
    print(f"  was_clamped: {result2['was_clamped']}")
    assert result2['was_clamped']
    assert not result2['gripper_open']
    
    # Case 3: Joint limit 근처
    adapter.set_current_state(joint_positions=np.deg2rad([349, 89, 0, 0, 0, 0]), gripper=1.0)
    action_edge = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.5])
    result3 = adapter.convert(action_edge)
    print(f"\n[Case 3] Near joint limit:")
    print(f"  targets (deg): {np.rad2deg(result3['joint_targets']).round(2)}")
    print(f"  was_clamped: {result3['was_clamped']}")
    
    print(f"\n✅ All tests passed! Clamp ratio: {result3['clamp_ratio']:.1%}")


if __name__ == "__main__":
    test_adapter()
