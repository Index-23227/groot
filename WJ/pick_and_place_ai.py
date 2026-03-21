#!/usr/bin/env python3
"""
AI Pick and Place - 자연어 명령으로 로봇 제어
Groot Vision (Gemini) + 캘리브레이션 + Doosan ROS2 제어

사용법:
  python3 ~/pick_and_place_ai.py

  명령 예시:
    "사이다를 오른쪽으로 옮겨줘"
    "빨간 블럭을 왼쪽에 놓아줘"
    "데자와를 앞으로 옮겨"
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess
import os
import sys
import time
import json
import base64
from io import BytesIO
from PIL import Image

os.environ["QT_QPA_PLATFORM"] = "xcb"

# === 캘리브레이션 데이터 (24개 포인트) ===
camera_points = np.array([
    [207.3, -201.0, 710.0],
    [2.4, -111.9, 657.0],
    [-105.9, -241.9, 734.0],
    [-147.5, -153.5, 681.0],
    [127.3, -86.5, 644.0],
    [-198.6, -17.0, 605.0],
    [211.9, 18.5, 586.0],
    [129.1, -112.9, 598.0],
    [-73.5, -47.8, 563.0],
    [84.1, -269.8, 683.0],
    [86.2, -267.1, 685.0],
    [-99.3, -91.0, 586.0],
    [81.9, -261.9, 710.0],
    [-99.8, -80.0, 606.0],
    [84.2, -244.7, 731.0],
    [-100.8, -65.5, 630.0],
    [77.8, -230.3, 763.0],
    [-99.6, -49.3, 661.0],
    [77.8, -226.6, 775.0],
    [-100.0, -45.8, 671.0],
    [77.9, -217.2, 776.0],
    [-99.9, -38.9, 677.0],
    [78.3, -214.2, 792.0],
    [-99.4, -36.3, 688.0],
], dtype=np.float64)

robot_points = np.array([
    [400.0, 200.0, 200.0],
    [500.0, 0.0, 200.0],
    [350.0, -100.0, 200.0],
    [450.0, -150.0, 200.0],
    [530.0, 120.0, 200.0],
    [600.0, -200.0, 200.0],
    [650.0, 200.0, 200.0],
    [530.0, 120.0, 260.0],
    [600.0, -80.0, 260.0],
    [350.0, 80.0, 260.0],
    [350.0, 80.0, 260.0],
    [550.0, -100.0, 260.0],
    [350.0, 80.0, 220.0],
    [550.0, -100.0, 220.0],
    [350.0, 80.0, 200.0],
    [550.0, -100.0, 200.0],
    [350.0, 80.0, 180.0],
    [550.0, -100.0, 180.0],
    [350.0, 80.0, 170.0],
    [550.0, -100.0, 170.0],
    [350.0, 80.0, 160.0],
    [550.0, -100.0, 160.0],
    [350.0, 80.0, 150.0],
    [550.0, -100.0, 150.0],
], dtype=np.float64)

# === 설정 ===
CAN_Z_THRESHOLD = 180
GRIP_CAN = 500
GRIP_BLOCK = 550
SAFE_Z = 400
MIN_Z = 120
MAX_REACH = 800

current_pos = [453.0, 0.0, 400.0]


# === 캘리브레이션 ===
def compute_transform(cam_pts, rob_pts):
    n = cam_pts.shape[0]
    A = np.hstack([cam_pts, np.ones((n, 1))])
    T, _, _, _ = np.linalg.lstsq(A, rob_pts, rcond=None)
    return T


def cam_to_robot(cam_xyz, T):
    cam_h = np.append(cam_xyz, 1.0)
    return cam_h @ T


# === 로봇 제어 ===
def move_line(x, y, z):
    if z < MIN_Z:
        z = MIN_Z
    cmd = (
        f'ros2 service call /dsr01/motion/move_line dsr_msgs2/srv/MoveLine '
        f'"{{pos: [{x:.1f}, {y:.1f}, {z:.1f}, 3.4, -180.0, 93.4], '
        f'vel: [80.0, 80.0], acc: [80.0, 80.0]}}"'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    time.sleep(0.5)


def move_robot(x, y, z):
    if z < MIN_Z:
        z = MIN_Z
    print(f"  ↑ z축 상승 (z={SAFE_Z}mm)")
    move_line(current_pos[0], current_pos[1], SAFE_Z)
    current_pos[2] = SAFE_Z

    print(f"  → xy 이동 (x={x:.1f}, y={y:.1f})")
    move_line(x, y, SAFE_Z)
    current_pos[0] = x
    current_pos[1] = y

    print(f"  ↓ z축 하강 (z={z:.1f}mm)")
    move_line(x, y, z)
    current_pos[2] = z


def gripper_control(position):
    position = max(0, min(700, position))
    cmd = (
        f'ros2 topic pub /dsr01/gripper/position_cmd std_msgs/msg/Int32 '
        f'"{{data: {position}}}" --once'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)


def go_home():
    print("  홈 위치로 복귀...")
    cmd = (
        'ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint '
        '"{pos: [0.0, 0.0, 90.0, 0.0, 90.0, 90.0], vel: 30.0, acc: 30.0}"'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)
    current_pos[0] = 453.0
    current_pos[1] = 0.0
    current_pos[2] = 400.0


# === AI Vision (Gemini) ===
def detect_with_ai(rgb_image, depth_frame, intrinsics, instruction):
    """Gemini로 물체 위치 감지 → 카메라 3D 좌표 반환"""
    try:
        import google.genai as genai

        # 이미지를 base64로 변환
        img_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        img_pil.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()

        client = genai.Client()

        prompt = f"""이 이미지에서 "{instruction}"에 해당하는 물체를 찾아주세요.

이미지 크기: {rgb_image.shape[1]}x{rgb_image.shape[0]} 픽셀

반드시 아래 JSON 형식으로만 답해주세요:
{{
  "pick_object": "물체 이름",
  "pick_px": 픽셀x좌표(정수),
  "pick_py": 픽셀y좌표(정수),
  "place_description": "놓을 위치 설명",
  "place_px": 놓을위치_픽셀x좌표(정수),
  "place_py": 놓을위치_픽셀y좌표(정수),
  "confidence": 신뢰도(0~1)
}}

주의:
- pick 좌표는 물체의 중심 픽셀 좌표입니다
- place 좌표는 놓을 위치의 픽셀 좌표입니다
- 명령에 놓을 위치가 없으면 테이블 빈 공간을 place로 잡아주세요
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )

        text = response.text
        # JSON 추출
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])

        print(f"\n[AI] 감지: {result['pick_object']} (신뢰도: {result['confidence']})")
        print(f"  Pick 픽셀: ({result['pick_px']}, {result['pick_py']})")
        print(f"  Place: {result['place_description']} ({result['place_px']}, {result['place_py']})")

        # 픽셀 → 3D 카메라 좌표
        pick_depth = depth_frame.get_distance(result['pick_px'], result['pick_py'])
        place_depth = depth_frame.get_distance(result['place_px'], result['place_py'])

        pick_3d = None
        place_3d = None

        if pick_depth > 0:
            pt = rs.rs2_deproject_pixel_to_point(
                intrinsics, [result['pick_px'], result['pick_py']], pick_depth)
            pick_3d = np.array([pt[0]*1000, pt[1]*1000, pt[2]*1000])
            print(f"  Pick 카메라 좌표: x={pick_3d[0]:.1f}, y={pick_3d[1]:.1f}, z={pick_3d[2]:.1f}")
        else:
            print(f"  Pick depth 없음!")

        if place_depth > 0:
            pt = rs.rs2_deproject_pixel_to_point(
                intrinsics, [result['place_px'], result['place_py']], place_depth)
            place_3d = np.array([pt[0]*1000, pt[1]*1000, pt[2]*1000])
            print(f"  Place 카메라 좌표: x={place_3d[0]:.1f}, y={place_3d[1]:.1f}, z={place_3d[2]:.1f}")
        else:
            print(f"  Place depth 없음!")

        return {
            "pick_object": result["pick_object"],
            "pick_cam": pick_3d,
            "place_cam": place_3d,
            "pick_px": (result["pick_px"], result["pick_py"]),
            "place_px": (result["place_px"], result["place_py"]),
            "confidence": result["confidence"],
        }

    except Exception as e:
        print(f"[AI Error] {e}")
        return None


# === 메인 파이프라인 ===
def run_pipeline(instruction, rgb_image, depth_frame, intrinsics, T):
    """자연어 명령 → AI 감지 → 로봇 실행"""

    # 1. AI로 물체 감지
    print(f"\n{'='*50}")
    print(f"  명령: {instruction}")
    print(f"{'='*50}")

    result = detect_with_ai(rgb_image, depth_frame, intrinsics, instruction)
    if result is None or result["pick_cam"] is None:
        print("물체 감지 실패!")
        return False

    # 2. 카메라 좌표 → 로봇 좌표
    pick_robot = cam_to_robot(result["pick_cam"], T)
    print(f"\n  Pick 로봇 좌표: x={pick_robot[0]:.1f}, y={pick_robot[1]:.1f}, z={pick_robot[2]:.1f}")

    place_robot = None
    if result["place_cam"] is not None:
        place_robot = cam_to_robot(result["place_cam"], T)
        print(f"  Place 로봇 좌표: x={place_robot[0]:.1f}, y={place_robot[1]:.1f}, z={place_robot[2]:.1f}")

    # 3. 도달 가능 확인
    pick_dist = np.sqrt(pick_robot[0]**2 + pick_robot[1]**2)
    if pick_dist > MAX_REACH:
        print(f"  도달 불가! 거리={pick_dist:.0f}mm")
        return False

    # 4. z 안전 제한
    if pick_robot[2] < MIN_Z:
        pick_robot[2] = MIN_Z

    # 5. 물체 타입 판별 (캔/블럭)
    if pick_robot[2] >= CAN_Z_THRESHOLD:
        obj_type, grip_val = "캔", GRIP_CAN
    else:
        obj_type, grip_val = "블럭", GRIP_BLOCK

    print(f"\n  물체 타입: {obj_type} → 그리퍼: {grip_val}")

    # 6. Pick 실행
    print(f"\n[1/5] Pick 이동")
    move_robot(pick_robot[0], pick_robot[1], pick_robot[2])

    print(f"\n[2/5] 그리퍼 잡기 ({grip_val})")
    gripper_control(grip_val)
    time.sleep(1)
    gripper_control(grip_val)
    time.sleep(1)

    # 7. Place 실행
    if place_robot is not None:
        place_z = place_robot[2]
        pick_z = pick_robot[2]
        place_z = place_z + (pick_z - MIN_Z) if place_z > MIN_Z else pick_z
        place_z += 20  # place 오프셋
        if obj_type == "블럭":
            place_z += 10
        if place_z < MIN_Z:
            place_z = MIN_Z

        place_dist = np.sqrt(place_robot[0]**2 + place_robot[1]**2)
        if place_dist > MAX_REACH:
            print(f"  Place 도달 불가! 거리={place_dist:.0f}mm")
            # 안전한 위치에 놓기
            place_robot[0] = 450.0
            place_robot[1] = 0.0
            place_z = 200.0

        print(f"\n[3/5] Place 이동 (x={place_robot[0]:.1f}, y={place_robot[1]:.1f}, z={place_z:.1f})")
        move_robot(place_robot[0], place_robot[1], place_z)

    print(f"\n[4/5] 그리퍼 열기")
    gripper_control(0)
    time.sleep(1)
    gripper_control(0)
    time.sleep(1)

    # 8. 홈 복귀
    print(f"\n[5/5] 홈 복귀")
    go_home()

    print(f"\n{'='*50}")
    print(f"  완료! {result['pick_object']} → {result.get('place_description', '목표 위치')}")
    print(f"{'='*50}\n")
    return True


# === 메인 ===
T = compute_transform(camera_points, robot_points)
print("=== AI Pick and Place ===")
avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"캘리브레이션 평균 오차: {avg_err:.1f}mm\n")

# 카메라 시작
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# auto exposure 안정화
for _ in range(30):
    pipeline.wait_for_frames()

print("카메라 준비 완료!")
print("명령어를 입력하세요 (q=종료):")
print("  예: 사이다를 오른쪽으로 옮겨줘")
print("  예: 빨간 블럭을 왼쪽에 놓아줘")
print("  예: 데자와를 집어서 앞에 놓아줘\n")

try:
    while True:
        instruction = input("\n명령> ").strip()
        if instruction.lower() == 'q':
            break
        if not instruction:
            continue

        # 최신 프레임 캡처
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            print("카메라 프레임 실패")
            continue

        rgb_image = np.asanyarray(color_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 실행
        success = run_pipeline(instruction, rgb_image, depth_frame, intrinsics, T)
        if not success:
            print("실행 실패. 다시 시도하세요.")

finally:
    pipeline.stop()
    print("\n종료")
