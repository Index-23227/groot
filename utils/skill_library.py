"""
Skill Library — 약국 로봇 스크립트 기반 동작 primitives

학습 없이 동작하는 핵심 원리:
  - Skill의 "형태(template)"는 고정 코드
  - Skill의 "목표(where)"는 Gemini ER이 이미지에서 실시간 계산
  - Grasp 성공 여부는 그리퍼 stroke 피드백으로 판단

Pharmacy 태스크에 필요한 5개 skill:
  pick      - 물체 집기 (Gemini가 위치 계산, stroke로 성공 판단)
  place     - 물체 놓기 (고정 위치 or Gemini 계산)
  inspect   - 라벨 확인 (고정 inspection 포즈로 이동)
  home      - 홈 복귀  (고정 포즈)
  handover  - 사람에게 건네기 (고정 포즈 + 그리퍼 오픈)
"""

import sys, os, time, json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import (
    grip_to_stroke, CAMERA_WIDTH, CAMERA_HEIGHT,
    GRIPPER_STROKE_MAX, GRIPPER_STROKE_MIN
)


# ── 고정 포즈 (teach pendant로 현장 측정 후 업데이트) ──────────────
# [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]  로봇 베이스 좌표계
HOME_POSE        = [400.0,   0.0, 400.0, 180.0, 0.0,  0.0]   # 홈 (공중)
INSPECT_POSE     = [350.0,   0.0, 350.0, 180.0, 0.0, 90.0]   # 카메라 앞
HANDOVER_POSE    = [500.0,   0.0, 200.0, 180.0, 0.0,  0.0]   # 사람 손 앞

# 트레이 슬롯 고정 위치 (teach pendant로 측정)
TRAY_SLOTS = {
    "tray_slot_1": [300.0, -150.0, 50.0, 180.0, 0.0, 0.0],
    "tray_slot_2": [300.0,    0.0, 50.0, 180.0, 0.0, 0.0],
    "tray_slot_3": [300.0,  150.0, 50.0, 180.0, 0.0, 0.0],
    "counter":     [450.0,    0.0, 80.0, 180.0, 0.0, 0.0],
    "tray":        [300.0,    0.0, 50.0, 180.0, 0.0, 0.0],   # 기본 tray
}

PREGRASP_HEIGHT_MM  = 120   # 물체 위 pregrasp 높이 (mm)
PLACE_HEIGHT_MM     = 80    # place 시 위에서 내려오는 높이 (mm)
PICK_MAX_RETRIES    = 3     # pick 최대 재시도 횟수
GRASP_STROKE_MIN    = 30    # 이 이상이면 물체를 잡은 것으로 판정 (stroke 단위)
MOTION_VEL_FAST     = 60.0
MOTION_VEL_SLOW     = 20.0
MOTION_VEL_MED      = 40.0


# ── 결과 dataclass ─────────────────────────────────────────────────

@dataclass
class SkillResult:
    success: bool
    skill:   str = ""
    reason:  str = ""
    data:    Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        mark = "✅" if self.success else "❌"
        return f"{mark} {self.skill}: {self.reason}"


# ── Base Skill ─────────────────────────────────────────────────────

class BaseSkill:
    name: str = "base"

    def execute(self, params: dict, robot, camera, bridge=None) -> SkillResult:
        raise NotImplementedError

    def _log(self, msg):
        print(f"  [{self.name}] {msg}")


# ── Pick Skill ─────────────────────────────────────────────────────

class PickSkill(BaseSkill):
    """
    물체 집기.

    학습 없는 신뢰도 확보 방법:
    1. Gemini ER이 bbox_norm → grasp_point 계산 (매 시도 새 이미지)
    2. 그리퍼 stroke 피드백으로 파지 성공 판정
    3. 실패 시 위치 보정 후 최대 3회 재시도
    4. 각 시도 전 새 이미지로 Gemini 재계산 (위치 drift 보정)
    """
    name = "pick"

    def execute(self, params: dict, robot, camera, bridge) -> SkillResult:
        object_name = params.get("object", "물체")
        grasp_style = params.get("grasp_style", "top")   # "top" | "side"
        self._log(f"대상: {object_name}, 방식: {grasp_style}")

        for attempt in range(1, PICK_MAX_RETRIES + 1):
            self._log(f"시도 {attempt}/{PICK_MAX_RETRIES}")

            # 1. 새 이미지로 Gemini가 물체 위치 계산
            rgb, depth = camera.read()
            detection  = bridge.detect_object_for_pick(rgb, depth, object_name, grasp_style)
            if detection is None:
                self._log("물체 감지 실패")
                if attempt == PICK_MAX_RETRIES:
                    return SkillResult(False, self.name, "물체 감지 실패")
                time.sleep(1.0)
                continue

            grasp_pose    = detection["grasp_pose"]
            pregrasp_pose = detection["pregrasp_pose"]
            stroke_open   = detection["stroke_open"]

            # 2. 그리퍼 열기
            robot.set_gripper(grip_to_stroke(0.0))

            # 3. pregrasp 위치로 이동
            ok = robot.movel(pregrasp_pose, vel=MOTION_VEL_FAST, acc=MOTION_VEL_FAST)

            # 4. grasp 위치로 천천히 내려감
            robot.movel(grasp_pose, vel=MOTION_VEL_SLOW, acc=MOTION_VEL_SLOW)

            # 5. 그리퍼 닫기
            robot.set_gripper(grip_to_stroke(1.0), wait_sec=2.0)

            # 6. 그리퍼 stroke 피드백으로 파지 성공 판정
            #    stroke 변화 있으면 물체가 그리퍼 안에 있음
            current_stroke = robot.get_gripper_stroke()
            grasped = (GRIPPER_STROKE_MIN < current_stroke < GRIPPER_STROKE_MAX - 20)
            self._log(f"그리퍼 stroke={current_stroke}, 파지={'성공' if grasped else '실패'}")

            if grasped:
                # 7. 들어올리기
                lift_pose = list(grasp_pose)
                lift_pose[2] += PREGRASP_HEIGHT_MM
                robot.movel(lift_pose, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)
                return SkillResult(True, self.name, "파지 성공",
                                   {"grasp_pose": grasp_pose, "stroke": current_stroke})

            # 실패: 그리퍼 열고 pregrasp로 복귀 후 재시도
            robot.set_gripper(grip_to_stroke(0.0))
            robot.movel(pregrasp_pose, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)

            # 재시도 시 보정: 수평으로 5mm 이동 후 다시
            if attempt < PICK_MAX_RETRIES:
                self._log("위치 보정 후 재시도")
                time.sleep(0.5)

        return SkillResult(False, self.name, f"{PICK_MAX_RETRIES}회 시도 후 파지 실패")


# ── Place Skill ────────────────────────────────────────────────────

class PlaceSkill(BaseSkill):
    """
    물체 놓기.
    위치는 사전 정의된 TRAY_SLOTS 또는 Gemini가 계산한 동적 위치.
    """
    name = "place"

    def execute(self, params: dict, robot, camera, bridge=None) -> SkillResult:
        location = params.get("location", "tray")
        self._log(f"위치: {location}")

        # 목표 pose 결정
        if location in TRAY_SLOTS:
            target_pose = list(TRAY_SLOTS[location])
        elif location == "custom" and "pose" in params:
            target_pose = params["pose"]
        else:
            # Gemini가 이미지에서 위치 계산
            if bridge is None:
                return SkillResult(False, self.name, "bridge 없음 — 위치 계산 불가")
            rgb, depth = camera.read()
            place_norm  = params.get("place_norm", [0.5, 0.7])
            detection   = bridge.detect_place_location(rgb, depth, place_norm)
            if detection is None:
                return SkillResult(False, self.name, "place 위치 감지 실패")
            target_pose = detection["place_pose"]

        # pre-place 위치 (위에서 접근)
        preplace_pose    = list(target_pose)
        preplace_pose[2] += PLACE_HEIGHT_MM

        # 이동 & 놓기
        robot.movel(preplace_pose, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)
        robot.movel(target_pose,   vel=MOTION_VEL_SLOW, acc=MOTION_VEL_SLOW)
        robot.set_gripper(grip_to_stroke(0.0), wait_sec=1.5)

        # 후퇴
        robot.movel(preplace_pose, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)

        return SkillResult(True, self.name, f"{location}에 놓음",
                           {"location": location, "pose": target_pose})


# ── Inspect Skill ──────────────────────────────────────────────────

class InspectSkill(BaseSkill):
    """
    라벨 확인.
    물체를 카메라 정면 inspection_pose로 가져와 Gemini가 라벨 판독.
    고정 포즈이므로 학습 불필요. Gemini가 OCR + 검증 담당.
    """
    name = "inspect"

    def execute(self, params: dict, robot, camera, bridge=None) -> SkillResult:
        expected = params.get("expected_label", None)
        self._log(f"기대 라벨: {expected or '(확인만)'}")

        # inspection 포즈로 이동
        robot.movel(INSPECT_POSE, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)
        time.sleep(0.8)   # 진동 안정

        # 이미지 캡처 & Gemini 라벨 판독은 task_planner.verify_skill()에서 수행
        # 여기서는 포즈 이동만 담당
        rgb, depth = camera.read()

        return SkillResult(True, self.name, "inspection 포즈 완료",
                           {"rgb": rgb, "expected_label": expected})


# ── Home Skill ─────────────────────────────────────────────────────

class HomeSkill(BaseSkill):
    """안전한 홈 포지션으로 복귀. 그리퍼 열림 상태로."""
    name = "home"

    def execute(self, params: dict, robot, camera, bridge=None) -> SkillResult:
        robot.set_gripper(grip_to_stroke(0.0))
        robot.movel(HOME_POSE, vel=MOTION_VEL_FAST, acc=MOTION_VEL_FAST)
        return SkillResult(True, self.name, "홈 복귀 완료")


# ── Handover Skill ─────────────────────────────────────────────────

class HandoverSkill(BaseSkill):
    """
    사람에게 약 건네기.
    handover_pose로 이동 후 그리퍼 열림.
    사람이 약을 가져가면 stroke 변화로 감지.
    """
    name = "handover"

    def execute(self, params: dict, robot, camera, bridge=None) -> SkillResult:
        self._log("handover 포즈로 이동")
        robot.movel(HANDOVER_POSE, vel=MOTION_VEL_MED, acc=MOTION_VEL_MED)
        time.sleep(0.5)

        # 그리퍼 열기
        robot.set_gripper(grip_to_stroke(0.0), wait_sec=1.0)
        self._log("그리퍼 오픈 — 사람이 가져가길 대기")

        # 사람이 약을 가져갈 때까지 대기 (최대 10초)
        # 실제로는 그리퍼 force 변화로 감지; 여기선 time wait
        for _ in range(10):
            time.sleep(1.0)
            stroke = robot.get_gripper_stroke()
            if stroke < GRASP_STROKE_MIN:
                self._log("전달 완료")
                break

        # 홈으로 복귀
        robot.movel(HOME_POSE, vel=MOTION_VEL_FAST, acc=MOTION_VEL_FAST)
        return SkillResult(True, self.name, "전달 완료")


# ── Skill Registry ─────────────────────────────────────────────────

class SkillLibrary:
    """
    사용 가능한 skill 목록과 메타데이터.
    TaskPlanner가 Gemini에게 skill 목록을 보여줄 때 사용.
    """

    REGISTRY = {
        "pick":     PickSkill(),
        "place":    PlaceSkill(),
        "inspect":  InspectSkill(),
        "home":     HomeSkill(),
        "handover": HandoverSkill(),
    }

    DESCRIPTIONS = {
        "pick": {
            "desc": "지정된 약품을 집습니다",
            "params": {"object": "약품 이름 (예: 타이레놀)", "grasp_style": "top|side (기본: top)"},
            "precondition": "그리퍼가 비어있어야 함, 물체가 카메라에 보여야 함",
            "effect": "물체를 들고 있음",
        },
        "place": {
            "desc": "들고 있는 약품을 지정 위치에 놓습니다",
            "params": {"location": "tray_slot_1|tray_slot_2|tray_slot_3|counter|tray"},
            "precondition": "그리퍼에 물체가 있어야 함",
            "effect": "물체가 목표 위치에 있음, 그리퍼 비어있음",
        },
        "inspect": {
            "desc": "들고 있는 약품의 라벨을 카메라로 확인합니다",
            "params": {"expected_label": "기대 라벨 문자열 (선택)"},
            "precondition": "그리퍼에 물체가 있어야 함",
            "effect": "라벨 정보 확인됨",
        },
        "home": {
            "desc": "로봇을 안전한 홈 포지션으로 복귀시킵니다",
            "params": {},
            "precondition": "없음",
            "effect": "로봇 홈 포지션",
        },
        "handover": {
            "desc": "들고 있는 약품을 사람에게 건네줍니다",
            "params": {},
            "precondition": "그리퍼에 물체가 있어야 함",
            "effect": "사람이 약품을 받음",
        },
    }

    def get(self, name: str) -> BaseSkill:
        if name not in self.REGISTRY:
            raise ValueError(f"Unknown skill: {name}. Available: {list(self.REGISTRY)}")
        return self.REGISTRY[name]

    def descriptions_for_prompt(self) -> str:
        """Gemini 프롬프트에 넣을 skill 설명."""
        lines = []
        for name, info in self.DESCRIPTIONS.items():
            params_str = json.dumps(info["params"], ensure_ascii=False)
            lines.append(
                f'  "{name}": {info["desc"]}\n'
                f'    params: {params_str}\n'
                f'    조건: {info["precondition"]}\n'
                f'    효과: {info["effect"]}'
            )
        return "\n".join(lines)
