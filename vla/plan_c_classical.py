"""
Plan C: Classical Pick & Place — Claude Vision (물체 인식) + cuRobo (IK/Motion Planning)
VLA 전부 실패 시 최후의 보루. 여전히 AI(Claude Vision) 사용.

사용법:
  python utils/plan_c_classical.py --instruction "파란 약병을 트레이에 옮겨줘"
  python utils/plan_c_classical.py --test-vision  (Claude Vision만 테스트)
"""

import os, sys, time, argparse, base64, json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *


def detect_object(image, instruction):
    """Claude Vision으로 물체 위치 추정"""
    try:
        from anthropic import Anthropic
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(image).save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        client = Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": f'이미지에서 "{instruction}"에 해당하는 물체의 위치를 찾아주세요.\n'
                 'JSON만 답해주세요: {{"object": "...", "px": 320, "py": 240, "conf": 0.9}}'},
            ]}],
        )
        text = resp.content[0].text
        result = json.loads(text[text.find("{"):text.rfind("}") + 1])
        print(f"[Vision] {result['object']} at ({result['px']}, {result['py']})")
        return result
    except Exception as e:
        print(f"[Vision] Error: {e}")
        return {"px": CAMERA_WIDTH // 2, "py": CAMERA_HEIGHT // 2, "conf": 0.0}


def pixel_to_robot(px, py):
    """픽셀 → 로봇 좌표 (현장 캘리브레이션 필요)"""
    # TODO: 현장에서 4점 캘리브레이션 후 업데이트
    x = 0.4 + (py - CAMERA_HEIGHT / 2) * 0.001
    y = (px - CAMERA_WIDTH / 2) * 0.001
    z = 0.05
    return np.array([x, y, z])


def plan_trajectory(pick_pos, place_pos):
    """cuRobo 또는 하드코딩 waypoint로 궤적 생성"""
    try:
        from curobo.wrap.reacher.ik_solver import IKSolver
        print("[cuRobo] Planning...")
        raise ImportError("TODO: cuRobo 연결")
    except ImportError:
        print("[Fallback] Hardcoded waypoints (현장에서 teach pendant로 측정)")
        home = np.deg2rad([0, 0, -90, 0, 90, 0])
        pre_pick = np.deg2rad([0, -30, -60, 0, 90, 0])
        pick = np.deg2rad([0, -45, -45, 0, 90, 0])
        pre_place = np.deg2rad([45, -30, -60, 0, 90, 0])
        place_j = np.deg2rad([45, -45, -45, 0, 90, 0])
        return [
            ("approach", pre_pick, False),
            ("pick", pick, False),
            ("grasp", pick, True),
            ("lift", pre_pick, True),
            ("move", pre_place, True),
            ("place", place_j, True),
            ("release", place_j, False),
            ("home", home, False),
        ]


def execute(trajectory):
    for name, target, grip in trajectory:
        deg = np.rad2deg(target)
        g = "CLOSE" if grip else "OPEN"
        print(f"  {name}: [{', '.join(f'{d:.0f}' for d in deg)}] {g}")
        time.sleep(0.5)  # TODO: 실제 servoj 명령 + 완료 대기
    print("✅ Done")


def main(args):
    if args.test_vision:
        img = np.random.randint(0, 255, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        detect_object(img, args.instruction)
        return

    from utils.doosan_recorder import CameraCapture
    camera = CameraCapture()

    while True:
        input(f"\n물체 배치 후 Enter... (instruction: {args.instruction})")
        img = camera.read()
        result = detect_object(img, args.instruction)
        pick = pixel_to_robot(result["px"], result["py"])
        place = np.array([0.3, 0.3, 0.05])  # TODO: 현장 설정
        traj = plan_trajectory(pick, place)
        execute(traj)
        if input("다시? [Enter/q]: ").strip().lower() == "q":
            break
    camera.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="파란 약병을 트레이에 옮겨줘")
    p.add_argument("--test-vision", action="store_true")
    main(p.parse_args())
