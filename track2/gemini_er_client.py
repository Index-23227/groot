"""
Gemini API 클라이언트 — Robotics-ER / 2.5 Flash / 2.5 Pro
- Object Pointing (pick/place 좌표)
- Task Decomposition (복합 명령 분해)
- Progress Estimation (성공 여부 판단)
- Failure Replanning (실패 시 재계획)
- In-Context Learning (few-shot waypoint 예제 기반 action 예측)

환경변수:
  GEMINI_API_KEY=your-key-here
  GEMINI_MODEL=gemini-2.5-flash-preview-05-20  (기본)
"""

import os
import json
import re
import cv2
import numpy as np
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# 모델 선택 (우선순위)
# 1. gemini-robotics-er-1.5  (preview, SOTA embodied reasoning)
# 2. gemini-2.5-flash         (범용, 빠름, 무료 tier: 분당 15 req)
# 3. gemini-2.5-pro           (pointing 정밀도 높음, 느림)
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")


def _get_model():
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.GenerativeModel(MODEL_NAME)


def _bgr_to_pil(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return json.loads(text)


# =============================================================
# Pointing: pick / place 좌표
# =============================================================

def get_pick_point(image_bgr, instruction):
    """
    이미지에서 pick 대상 물체의 grasp point (pixel x, y) 반환.
    프롬프트: 물체 시각적 특징 + gripper 유형 + grasp point 위치 명시.
    """
    model = _get_model()
    h, w = image_bgr.shape[:2]

    prompt = f"""You are a robot vision system with a parallel gripper.
Point at the CENTER of the target object's graspable region for:
"{instruction}"

The grasp point should be at the MIDDLE of the object's body where
a parallel gripper can securely grip it (not the top/cap, not the bottom).

Return ONLY JSON: {{"x": <int>, "y": <int>}}
Image resolution: {w}x{h}. Do not include any other text."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        result = _parse_json(resp.text)
        x = max(0, min(w - 1, int(result["x"])))
        y = max(0, min(h - 1, int(result["y"])))
        return (x, y)
    except Exception as e:
        print(f"[Gemini] get_pick_point error: {e}")
        return None


def get_place_point(image_bgr, instruction):
    """이미지에서 place 목표 위치의 pixel (x, y) 반환."""
    model = _get_model()
    h, w = image_bgr.shape[:2]

    prompt = f"""You are a robot vision system.
Find the target location to PLACE an object for:
"{instruction}"

The point should be the CENTER of the target receptacle or area
where the object should be put down.

Return ONLY JSON: {{"x": <int>, "y": <int>}}
Image resolution: {w}x{h}."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        result = _parse_json(resp.text)
        x = max(0, min(w - 1, int(result["x"])))
        y = max(0, min(h - 1, int(result["y"])))
        return (x, y)
    except Exception as e:
        print(f"[Gemini] get_place_point error: {e}")
        return None


# =============================================================
# Task Decomposition
# =============================================================

def decompose_task(image_bgr, instruction):
    """
    복합 명령을 pick/place step 리스트로 분해.
    Objects는 시각적 특징(색, 크기, 위치)으로, location은 공간적 특징으로 기술.
    """
    model = _get_model()

    prompt = f"""You are a robot task planner. A 6-DOF robot arm with a parallel gripper is in the scene.

Instruction: "{instruction}"

Looking at the image, break this into sequential pick-and-place steps.
Each step is either "pick <object>" or "place <location>".
Objects should be described by visual features (color, shape, size, position).
Locations should be described by spatial features (left tray, right area, etc).

Return ONLY a JSON array:
[
  {{"action": "pick", "target": "the red medicine bottle on the left"}},
  {{"action": "place", "target": "the tray on the far left"}},
  ...
]"""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] decompose_task error: {e}")
        return None


# =============================================================
# Progress Check
# =============================================================

def check_progress(image_bgr, step_description):
    """
    현재 step이 완료되었는지 판단.
    Returns: {"complete": bool, "reason": str}
    """
    model = _get_model()

    prompt = f"""You are monitoring a robot task.
The robot just attempted: "{step_description}"

Look at the current image carefully:
- Is the target object now in the correct location?
- Or is it still on the table / dropped / misplaced?

Return ONLY JSON: {{"complete": true/false, "reason": "brief explanation"}}"""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] check_progress error: {e}")
        return {"complete": False, "reason": f"error: {e}"}


# =============================================================
# Failure Replanning (실패 시 재계획)
# =============================================================

def replan_after_failure(image_bgr, failure_description):
    """
    Grasp/place 실패 후 Gemini에게 실패 이미지를 보여주고 재계획 요청.
    VLA에서는 불가능한 기능 — Gemini는 자연어로 실패 원인 분석 + 수정안 제시.

    Returns:
        {"adjustment": str, "new_pick_point": {"x": int, "y": int}} or None
    """
    model = _get_model()
    h, w = image_bgr.shape[:2]

    prompt = f"""The robot just failed: {failure_description}

Looking at the current image:
1. Where is the target object now?
2. What likely went wrong? (offset, wrong object, collision, slipped?)
3. How should the robot adjust its approach?

Return ONLY JSON:
{{
  "adjustment": "brief description of what to change",
  "new_pick_point": {{"x": <int>, "y": <int>}}
}}
Image resolution: {w}x{h}."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] replan error: {e}")
        return None


# =============================================================
# In-Context Learning (Few-shot Action Prediction)
# =============================================================

def predict_next_action_icl(image_bgr, current_tcp, instruction, examples=None):
    """
    Few-shot In-Context Learning: 몇 개의 (image, tcp, action) 예시를
    context에 넣어서 다음 action을 예측.

    teach pendant로 5~10개 waypoint 사진만 찍으면 됨 (teleop 데모 아님).

    Args:
        image_bgr: 현재 카메라 이미지
        current_tcp: [x, y, z, rx, ry, rz] 현재 TCP pose (mm, deg)
        instruction: 자연어 명령
        examples: list of dicts, each with:
            - "image_path": str (예시 이미지 경로)
            - "tcp": [x, y, z, rx, ry, rz]
            - "action": str (예: "move_to(400, 0, 250, 0, 180, 0)")
            - "description": str (왜 이 행동을 하는지)

    Returns:
        {"action": "function_call_string", "reasoning": "explanation"}
    """
    model = _get_model()

    # Context 구성
    contents = []

    system_text = f"""You are controlling a Doosan E0509 6-DOF robot arm with an RH-P12-RN-A parallel gripper.

Available robot API functions:
1. move_to(x, y, z, rx=0, ry=180, rz=0) — Move TCP to cartesian position (mm, degree). Default: gripper pointing down.
2. gripper_open() — Open the gripper.
3. gripper_close() — Close the gripper.
4. go_home() — Move to home position [0,0,0,0,0,0].

Task: "{instruction}"
"""
    contents.append(system_text)

    # Few-shot examples
    if examples:
        contents.append("\nHere are examples of this robot performing similar tasks:\n")
        for i, ex in enumerate(examples):
            contents.append(f"Example {i+1}:")
            # 이미지 첨부
            if os.path.exists(ex.get("image_path", "")):
                ex_img = cv2.imread(ex["image_path"])
                if ex_img is not None:
                    contents.append(_bgr_to_pil(ex_img))
            tcp_str = ", ".join(f"{v:.1f}" for v in ex["tcp"])
            contents.append(
                f"Current TCP: [{tcp_str}]\n"
                f"Action: {ex['action']}\n"
                f"Reasoning: {ex['description']}\n"
            )

    # 현재 상태
    contents.append("\nNow, given the current scene:")
    contents.append(_bgr_to_pil(image_bgr))
    tcp_str = ", ".join(f"{v:.1f}" for v in current_tcp)
    contents.append(
        f"Current TCP: [{tcp_str}]\n"
        f"Task: \"{instruction}\"\n\n"
        f"What action should the robot take next?\n"
        f"Return ONLY JSON: {{\"action\": \"function_call\", \"reasoning\": \"explanation\"}}"
    )

    try:
        resp = model.generate_content(contents)
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] ICL predict error: {e}")
        return None


def run_icl_episode(image_reader, tcp_reader, action_executor,
                    instruction, examples=None, max_steps=20):
    """
    ICL 모드로 전체 에피소드 실행.

    Args:
        image_reader: callable() -> BGR image
        tcp_reader: callable() -> [x,y,z,rx,ry,rz]
        action_executor: callable(action_string) -> bool
        instruction: 자연어 명령
        examples: ICL 예시 리스트
        max_steps: 최대 step 수
    """
    print(f"\n[ICL] Running episode: {instruction}")
    print(f"[ICL] {len(examples or [])} examples in context\n")

    for step in range(max_steps):
        image = image_reader()
        tcp = tcp_reader()

        result = predict_next_action_icl(
            image, tcp, instruction, examples)
        if result is None:
            print(f"[ICL] Step {step+1}: prediction failed")
            continue

        action = result.get("action", "")
        reasoning = result.get("reasoning", "")
        print(f"[ICL] Step {step+1}: {action}")
        print(f"       Reason: {reasoning}")

        if "go_home" in action:
            action_executor(action)
            print("[ICL] Episode complete (go_home called)")
            break

        action_executor(action)

    print(f"[ICL] Episode finished ({step+1} steps)")


# =============================================================
# Test
# =============================================================

if __name__ == "__main__":
    import sys

    if not GEMINI_KEY:
        print("Set GEMINI_API_KEY first!")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Camera read failed")
        sys.exit(1)

    # Pointing test
    point = get_pick_point(frame, "the bottle")
    print(f"Pick point: {point}")

    if point:
        vis = frame.copy()
        cv2.circle(vis, point, 10, (0, 255, 0), 2)
        cv2.imwrite("gemini_test.jpg", vis)
        print("Saved gemini_test.jpg")
