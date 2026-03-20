"""
Gemini Robotics-ER 1.5 API 클라이언트
- Object Pointing (pick/place 좌표)
- Task Decomposition (복합 명령 분해)
- Progress Estimation (성공 여부 판단)

환경변수:
  GEMINI_API_KEY=your-key-here
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
# 1. gemini-robotics-er-1.5  (preview, 최고 성능)
# 2. gemini-2.5-flash         (범용, 빠름)
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


def get_pick_point(image_bgr, instruction):
    """
    이미지에서 pick 대상 물체의 grasp point (pixel x, y) 반환.

    Returns:
        (x, y) tuple 또는 None
    """
    model = _get_model()
    h, w = image_bgr.shape[:2]

    prompt = f"""You are a robot vision system. A 6-DOF robot arm with a parallel gripper is in the scene.
Given the image, find the best grasp point for:
"{instruction}"

Return ONLY a JSON object with the pixel coordinates of the CENTER of the object where the gripper should close:
{{"x": <int>, "y": <int>}}

The point should be at the center of the object's graspable region.
Image resolution is {w}x{h}. Do not include any other text."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        result = _parse_json(resp.text)
        x, y = int(result["x"]), int(result["y"])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return (x, y)
    except Exception as e:
        print(f"[Gemini] get_pick_point error: {e}")
        return None


def get_place_point(image_bgr, instruction):
    """
    이미지에서 place 목표 위치의 pixel (x, y) 반환.
    """
    model = _get_model()
    h, w = image_bgr.shape[:2]

    prompt = f"""You are a robot vision system.
Given the image, find the target location to PLACE an object for:
"{instruction}"

Return ONLY a JSON object:
{{"x": <int>, "y": <int>}}

The point should be the center of the target receptacle or area.
Image resolution is {w}x{h}."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        result = _parse_json(resp.text)
        x, y = int(result["x"]), int(result["y"])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return (x, y)
    except Exception as e:
        print(f"[Gemini] get_place_point error: {e}")
        return None


def decompose_task(image_bgr, instruction):
    """
    복합 명령을 pick/place step 리스트로 분해.

    Returns:
        [{"action": "pick", "target": "..."}, {"action": "place", "target": "..."}, ...]
    """
    model = _get_model()

    prompt = f"""You are a robot task planner. A 6-DOF robot arm with a parallel gripper is in the scene.

Instruction: "{instruction}"

Break this into sequential pick-and-place steps.
Return ONLY a JSON array:
[
  {{"action": "pick", "target": "object description"}},
  {{"action": "place", "target": "location description"}},
  ...
]

Keep descriptions concise and visually identifiable (color, shape, location)."""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] decompose_task error: {e}")
        return None


def check_progress(image_bgr, step_description):
    """
    현재 step이 완료되었는지 판단.

    Returns:
        {"complete": bool, "reason": str}
    """
    model = _get_model()

    prompt = f"""You are monitoring a robot task.
The current step is: "{step_description}"

Look at the image. Is this step COMPLETE?
Return ONLY JSON:
{{"complete": true/false, "reason": "brief explanation"}}"""

    try:
        resp = model.generate_content([_bgr_to_pil(image_bgr), prompt])
        return _parse_json(resp.text)
    except Exception as e:
        print(f"[Gemini] check_progress error: {e}")
        return {"complete": False, "reason": f"error: {e}"}


# ===== 테스트 =====
if __name__ == "__main__":
    import sys

    if not GEMINI_KEY:
        print("Set GEMINI_API_KEY first!")
        sys.exit(1)

    # 카메라에서 1프레임 캡처
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Camera read failed")
        sys.exit(1)

    # Pointing 테스트
    point = get_pick_point(frame, "the bottle")
    print(f"Pick point: {point}")

    if point:
        vis = frame.copy()
        cv2.circle(vis, point, 10, (0, 255, 0), 2)
        cv2.imwrite("gemini_test.jpg", vis)
        print("Saved gemini_test.jpg")
