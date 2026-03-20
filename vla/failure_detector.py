"""
실패 감지 & 자동 재시도 — Level 4

Joint 변화량, clamp 비율 등으로 grasp 실패를 감지하고
재시도 또는 classical fallback으로 전환.

사용법:
  controller에서 import하여 사용
"""

import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *


class FailureDetector:
    """실행 중 실패를 감지하는 모니터"""

    def __init__(self,
                 stall_threshold=0.002,      # rad — 이 이하면 "안 움직임"
                 stall_window=10,             # 연속 N스텝 stall이면 실패
                 clamp_ratio_threshold=0.8,   # clamp 비율 이 이상이 지속되면 위험
                 clamp_window=5,
                 max_retries=2):
        self.stall_threshold = stall_threshold
        self.stall_window = stall_window
        self.clamp_ratio_threshold = clamp_ratio_threshold
        self.clamp_window = clamp_window
        self.max_retries = max_retries
        self.reset()

    def reset(self):
        self._joint_history = []
        self._clamp_history = []
        self._retry_count = 0

    def update(self, joints: np.ndarray, clamp_ratio: float) -> dict:
        """매 스텝 호출. 실패 상태 반환."""
        self._joint_history.append(joints.copy())
        self._clamp_history.append(clamp_ratio)

        result = {
            "stalled": False,
            "over_clamped": False,
            "should_retry": False,
            "should_fallback": False,
        }

        # Stall 감지: 최근 N스텝 동안 joint 변화 없음
        if len(self._joint_history) >= self.stall_window:
            recent = np.array(self._joint_history[-self.stall_window:])
            deltas = np.abs(np.diff(recent, axis=0)).max(axis=1)
            if np.all(deltas < self.stall_threshold):
                result["stalled"] = True

        # 과도한 clamp 감지: 모델이 비현실적 action을 계속 출력
        if len(self._clamp_history) >= self.clamp_window:
            recent_clamp = self._clamp_history[-self.clamp_window:]
            if all(c > self.clamp_ratio_threshold for c in recent_clamp):
                result["over_clamped"] = True

        # 실패 시 재시도 판단
        if result["stalled"] or result["over_clamped"]:
            if self._retry_count < self.max_retries:
                result["should_retry"] = True
                self._retry_count += 1
            else:
                result["should_fallback"] = True

        return result

    @property
    def retry_count(self):
        return self._retry_count
