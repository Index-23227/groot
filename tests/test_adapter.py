"""
단위 테스트 — 로봇 없이 실행. 해커톤 전에 로직 검증.
  python tests/test_adapter.py
"""
import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.doosan_action_adapter import DoosanActionAdapter, DoosanSafetyConfig
from configs.doosan_e0509_config import *

def test_config():
    assert NUM_JOINTS == 6 and ACTION_DIM == 7 and len(JOINT_NAMES) == 6
    assert JOINT_NAMES[0] == "joint_1", f"Expected 'joint_1', got '{JOINT_NAMES[0]}'"
    assert GRIPPER_TYPE == "robotis_rh_p12_rn_a"
    assert GRIPPER_STROKE_MAX == 700
    print("✅ config OK")

def test_grip_conversion():
    """grip ↔ stroke 변환 테스트"""
    assert grip_to_stroke(0.0) == 0, "grip 0.0 → stroke 0 (open)"
    assert grip_to_stroke(1.0) == 700, "grip 1.0 → stroke 700 (close)"
    assert grip_to_stroke(0.5) == 350, "grip 0.5 → stroke 350"
    assert stroke_to_grip(0) == 0.0
    assert stroke_to_grip(700) == 1.0
    assert abs(stroke_to_grip(350) - 0.5) < 0.01
    # 클리핑
    assert grip_to_stroke(-0.5) == 0
    assert grip_to_stroke(1.5) == 700
    print("✅ grip conversion OK")

def test_normal():
    """grip=0.8 → 닫힘(close), grip=0.2 → 열림(open)"""
    a = DoosanActionAdapter()
    a.set_current_state(np.zeros(NUM_JOINTS), 0.0)
    # grip 0.8 → close (> 0.5)
    r = a.convert(np.array([0.01]*NUM_JOINTS + [0.8]))
    assert not r["was_clamped"]
    assert r["gripper_close"] == True, "grip 0.8 should be CLOSE"
    assert r["gripper_open"] == False
    assert r["gripper_stroke"] == 560
    # grip 0.2 → open (< 0.5)
    r = a.convert(np.array([0.01]*NUM_JOINTS + [0.2]))
    assert r["gripper_close"] == False, "grip 0.2 should be OPEN"
    assert r["gripper_open"] == True
    assert r["gripper_stroke"] == 140
    print("✅ normal OK")

def test_clamp():
    a = DoosanActionAdapter()
    a.set_current_state(np.zeros(NUM_JOINTS), 0.0)
    r = a.convert(np.array([0.5]*NUM_JOINTS + [0.3]))
    assert r["was_clamped"]
    assert r["gripper_close"] == False, "grip 0.3 should be OPEN"
    print("✅ clamp OK")

def test_limits():
    a = DoosanActionAdapter()
    a.set_current_state(np.deg2rad([349, 89, 0, 0, 0, 0]), 1.0)
    r = a.convert(np.array([0.05, 0.05, 0, 0, 0, 0, 0.5]))
    assert np.rad2deg(r["joint_targets_rad"][0]) <= 350.01
    assert np.rad2deg(r["joint_targets_rad"][1]) <= 95.01
    print("✅ limits OK")

def test_degree_conversion():
    """adapter가 radian → degree 변환을 정확히 하는지"""
    a = DoosanActionAdapter()
    a.set_current_state(np.deg2rad([10, 20, 30, 40, 50, 60]), 0.0)
    r = a.convert(np.array([0.0]*NUM_JOINTS + [0.5]))
    # delta=0이므로 target_deg ≈ [10, 20, 30, 40, 50, 60]
    np.testing.assert_array_almost_equal(r["joint_targets_deg"], [10, 20, 30, 40, 50, 60], decimal=1)
    print("✅ degree conversion OK")

def test_blender_first_chunk():
    """첫 chunk는 blending 없이 execute_horizon만큼 그대로 반환"""
    from utils.doosan_vla_controller import TemporalBlender
    b = TemporalBlender(execute_horizon=4, overlap=4, decay=0.7)
    chunk = np.arange(16*7, dtype=float).reshape(16, 7)
    result = b.blend(chunk)
    assert result.shape == (4, 7), f"Expected (4,7), got {result.shape}"
    np.testing.assert_array_equal(result, chunk[:4])
    print("✅ blender first_chunk OK")

def test_blender_smooth():
    """두 번째 chunk부터 blending이 적용되어 값이 바뀌어야 함"""
    from utils.doosan_vla_controller import TemporalBlender
    b = TemporalBlender(execute_horizon=4, overlap=4, decay=0.7)
    chunk1 = np.ones((16, 7)) * 1.0
    chunk2 = np.ones((16, 7)) * 2.0
    b.blend(chunk1)  # 첫 chunk: [1,1,1,...] → a0~a3 실행, a4~a15 잔여
    result = b.blend(chunk2)  # 두 번째: blending 적용
    # result[0]은 chunk1 잔여(1.0)와 chunk2(2.0)의 가중 평균 → 1.0과 2.0 사이
    assert 1.0 < result[0, 0] < 2.0, f"Expected blended value, got {result[0,0]}"
    # result 뒤쪽일수록 chunk2(2.0)에 가까워야 함
    assert result[-1, 0] > result[0, 0], "Later actions should be closer to new chunk"
    print("✅ blender smooth OK")

def test_blender_reset():
    """reset 후 첫 chunk처럼 동작해야 함"""
    from utils.doosan_vla_controller import TemporalBlender
    b = TemporalBlender(execute_horizon=4, overlap=4, decay=0.7)
    b.blend(np.ones((16, 7)))
    b.reset()
    chunk = np.ones((16, 7)) * 3.0
    result = b.blend(chunk)
    np.testing.assert_array_almost_equal(result, chunk[:4])
    print("✅ blender reset OK")

if __name__ == "__main__":
    test_config()
    test_grip_conversion()
    test_normal()
    test_clamp()
    test_limits()
    test_degree_conversion()
    test_blender_first_chunk()
    test_blender_smooth()
    test_blender_reset()
    print("\n✅ All tests passed!")
