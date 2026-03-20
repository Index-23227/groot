"""
Dual-Brain 모드: Gemini-ER (planner) + VLA (executor)

Gemini-ER: 무엇을 할지 결정 (task decomposition + progress check)
VLA:       어떻게 할지 실행 (end-to-end manipulation)

Usage:
  python dual_brain.py --instruction "약병들을 색깔별로 분류해" \
    --vla-url http://localhost:5555
"""

import argparse
import time
import cv2
import numpy as np

from gemini_er_client import decompose_task, check_progress

# VLA client는 Track 1 (루트 레포)에서 가져옴
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_dual_brain(instruction, vla_url, camera_index=0, max_retries=3):
    """
    Gemini-ER가 task를 분해하고, VLA가 각 step을 실행.
    매 step 후 Gemini-ER가 성공 여부를 판단.
    """
    cap = cv2.VideoCapture(camera_index)

    # 1. 첫 이미지로 task 분해
    ret, frame = cap.read()
    if not ret:
        print("Camera failed")
        return

    print(f"\n{'='*60}")
    print(f"  Dual-Brain: Gemini-ER + VLA")
    print(f"  Instruction: {instruction}")
    print(f"{'='*60}\n")

    steps = decompose_task(frame, instruction)
    if steps is None:
        print("[Gemini] Task decomposition failed!")
        cap.release()
        return

    print(f"[Gemini] Plan ({len(steps)} steps):")
    for i, s in enumerate(steps):
        print(f"  {i+1}. {s['action']} → {s['target']}")

    # 2. 각 step을 VLA로 실행
    for i, step in enumerate(steps):
        step_text = f"{step['action']} {step['target']}"
        print(f"\n{'─'*40}")
        print(f"  Step {i+1}/{len(steps)}: {step_text}")
        print(f"{'─'*40}")

        for attempt in range(max_retries):
            # VLA 실행
            print(f"  [VLA] Executing (attempt {attempt+1})...")
            success = call_vla(vla_url, step_text, camera_index)

            # Gemini-ER 확인
            time.sleep(1.0)
            ret, frame = cap.read()
            if ret:
                progress = check_progress(frame, step_text)
                if progress.get("complete", False):
                    print(f"  [Gemini] ✅ Step complete: {progress.get('reason','')}")
                    break
                else:
                    print(f"  [Gemini] ❌ Not complete: {progress.get('reason','')}")
            
            if attempt < max_retries - 1:
                print(f"  [Retry] Trying again...")
                time.sleep(1.0)
        else:
            print(f"  [FAIL] Step {i+1} failed after {max_retries} attempts")

    print(f"\n{'='*60}")
    print(f"  Dual-Brain pipeline complete!")
    print(f"{'='*60}")
    cap.release()


def call_vla(vla_url, instruction, camera_index):
    """VLA inference server에 1 에피소드 실행 요청"""
    import requests
    import base64
    from io import BytesIO
    from PIL import Image

    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    # 이미지 인코딩
    buf = BytesIO()
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = requests.post(
            f"{vla_url}/predict",
            json={"image": img_b64, "instruction": instruction, "state": [0]*7},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  [VLA] Error: {e}")
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--vla-url", default="http://localhost:5555")
    p.add_argument("--camera-index", type=int, default=0)
    args = p.parse_args()

    run_dual_brain(args.instruction, args.vla_url, args.camera_index)
