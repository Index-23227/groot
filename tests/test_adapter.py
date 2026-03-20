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
    print("✅ config OK")

def test_normal():
    a = DoosanActionAdapter()
    a.set_current_state(np.zeros(NUM_JOINTS), 0.0)
    r = a.convert(np.array([0.01]*NUM_JOINTS + [0.8]))
    assert not r["was_clamped"] and r["gripper_open"]
    print("✅ normal OK")

def test_clamp():
    a = DoosanActionAdapter()
    a.set_current_state(np.zeros(NUM_JOINTS), 0.0)
    r = a.convert(np.array([0.5]*NUM_JOINTS + [0.3]))
    assert r["was_clamped"] and not r["gripper_open"]
    print("✅ clamp OK")

def test_limits():
    a = DoosanActionAdapter()
    a.set_current_state(np.deg2rad([349, 89, 0, 0, 0, 0]), 1.0)
    r = a.convert(np.array([0.05, 0.05, 0, 0, 0, 0, 0.5]))
    assert np.rad2deg(r["joint_targets"][0]) <= 350.01
    assert np.rad2deg(r["joint_targets"][1]) <= 95.01
    print("✅ limits OK")

if __name__ == "__main__":
    test_config()
    test_normal()
    test_clamp()
    test_limits()
    print("\n✅ All tests passed!")
