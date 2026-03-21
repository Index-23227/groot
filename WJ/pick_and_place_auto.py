#!/usr/bin/env python3
"""
Pick and Place 자동화
- 카메라 클릭 → 물체 높이(z)로 캔/블럭 자동 구분
- 캔(높은 물체): 그리퍼 500
- 블럭(낮은 물체): 그리퍼 550
- 놓을 위치 클릭 → 이동 → 놓기 → 홈 복귀
- 연속 프레임 0.5초 간격 저장
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess
import os
import json
import time
from datetime import datetime

os.environ["QT_QPA_PLATFORM"] = "xcb"

# === 캘리브레이션 데이터 (20개 포인트) ===
camera_points = np.array([
    # 사이다 (z=190)
    [207.3, -201.0, 710.0],
    [2.4, -111.9, 657.0],
    [-105.9, -241.9, 734.0],
    [-147.5, -153.5, 681.0],
    [127.3, -86.5, 644.0],
    [-198.6, -17.0, 605.0],
    [211.9, 18.5, 586.0],
    # 얼박사 (z=240)
    [129.1, -112.9, 598.0],
    [-73.5, -47.8, 563.0],
    [84.1, -269.8, 683.0],
    [86.2, -267.1, 685.0],
    [-99.3, -91.0, 586.0],
    # 데자와 (z=210)
    [81.9, -261.9, 710.0],
    [-99.8, -80.0, 606.0],
    # 사이다 (z=190)
    [84.2, -244.7, 731.0],
    [-100.8, -65.5, 630.0],
    # 빨간 블럭 (z=170)
    [77.8, -230.3, 763.0],
    [-99.6, -49.3, 661.0],
    # 노란 블럭 (z=160)
    [77.8, -226.6, 775.0],
    [-100.0, -45.8, 671.0],
    # 초록 블럭 (z=150)
    [77.9, -217.2, 776.0],
    [-99.9, -38.9, 677.0],
    # 파란 블럭 (z=140)
    [78.3, -214.2, 792.0],
    [-99.4, -36.3, 688.0],
], dtype=np.float64)

robot_points = np.array([
    # 사이다
    [400.0, 200.0, 190.0],
    [500.0, 0.0, 190.0],
    [350.0, -100.0, 190.0],
    [450.0, -150.0, 190.0],
    [530.0, 120.0, 190.0],
    [600.0, -200.0, 190.0],
    [650.0, 200.0, 190.0],
    # 얼박사
    [530.0, 120.0, 240.0],
    [600.0, -80.0, 240.0],
    [350.0, 80.0, 240.0],
    [350.0, 80.0, 240.0],
    [550.0, -100.0, 240.0],
    # 데자와
    [350.0, 80.0, 210.0],
    [550.0, -100.0, 210.0],
    # 사이다
    [350.0, 80.0, 190.0],
    [550.0, -100.0, 190.0],
    # 빨간 블럭
    [350.0, 80.0, 170.0],
    [550.0, -100.0, 170.0],
    # 노란 블럭
    [350.0, 80.0, 160.0],
    [550.0, -100.0, 160.0],
    # 초록 블럭
    [350.0, 80.0, 150.0],
    [550.0, -100.0, 150.0],
    # 파란 블럭
    [350.0, 80.0, 140.0],
    [550.0, -100.0, 140.0],
], dtype=np.float64)

# === 물체 구분 기준 ===
# 로봇 z >= 180 → 캔 (사이다, 데자와, 얼박사) → 그리퍼 500
# 로봇 z < 180  → 블럭 → 그리퍼 550
CAN_Z_THRESHOLD = 180
GRIP_CAN = 500
GRIP_BLOCK = 550

# === 설정 ===
SAFE_Z = 400
MIN_Z = 120
MAX_REACH = 800
DATA_DIR = os.path.expanduser("~/pick_place_data")
FRAME_INTERVAL = 0.5

# === 상태 ===
STATE_PICK = "PICK"
STATE_PLACE = "PLACE"

current_state = STATE_PICK
current_pos = [453.0, 0.0, 300.0]
pick_z = 0
grip_value = 0
cycle_count = 0
cycle_data = {}
object_type = ""

# 프레임 저장
recording = False
frame_list = []


def compute_transform(cam_pts, rob_pts):
    n = cam_pts.shape[0]
    A = np.hstack([cam_pts, np.ones((n, 1))])
    T, _, _, _ = np.linalg.lstsq(A, rob_pts, rcond=None)
    return T


def cam_to_robot(cam_xyz, T):
    cam_h = np.append(cam_xyz, 1.0)
    return cam_h @ T


def move_line(x, y, z, rz=93.4):
    if z < MIN_Z:
        z = MIN_Z
    cmd = (
        f'ros2 service call /dsr01/motion/move_line dsr_msgs2/srv/MoveLine '
        f'"{{pos: [{x:.1f}, {y:.1f}, {z:.1f}, 3.4, -180.0, {rz}], '
        f'vel: [80.0, 80.0], acc: [80.0, 80.0]}}"'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)


def move_z_only(z):
    if z < MIN_Z:
        z = MIN_Z
    move_line(current_pos[0], current_pos[1], z)
    current_pos[2] = z


def move_xy_only(x, y):
    move_line(x, y, current_pos[2])
    current_pos[0] = x
    current_pos[1] = y


def move_robot(x, y, z):
    if z < MIN_Z:
        print(f"  안전 제한: z={z:.1f} → {MIN_Z}mm")
        z = MIN_Z

    print(f"  ↑ 상승 (z={SAFE_Z}mm)")
    move_z_only(SAFE_Z)

    print(f"  → XY 이동 (x={x:.1f}, y={y:.1f})")
    move_xy_only(x, y)

    print(f"  ↓ 하강 (z={z:.1f}mm)")
    move_z_only(z)

    print("  이동 완료!")


def gripper_control(position):
    position = max(0, min(700, position))
    cmd = (
        f'ros2 topic pub /dsr01/gripper/position_cmd std_msgs/msg/Int32 '
        f'"{{data: {position}}}" --once'
    )
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)


def check_reachable(x, y):
    dist = np.sqrt(x**2 + y**2)
    return dist <= MAX_REACH, dist


def classify_object(robot_z):
    """로봇 z값으로 캔/블럭 구분"""
    if robot_z >= CAN_Z_THRESHOLD:
        return "캔", GRIP_CAN
    else:
        return "블럭", GRIP_BLOCK


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


def save_frame(img, timestamp):
    frame_list.append({
        "timestamp": timestamp,
        "robot_pos": list(current_pos),
        "image": img.copy(),
    })


def save_cycle(cycle_num, data, frames):
    cycle_dir = os.path.join(DATA_DIR, f"cycle_{cycle_num:04d}")
    frames_dir = os.path.join(cycle_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if "pick_frame" in data and data["pick_frame"] is not None:
        cv2.imwrite(os.path.join(cycle_dir, "pick_frame.jpg"), data["pick_frame"])
    if "place_frame" in data and data["place_frame"] is not None:
        cv2.imwrite(os.path.join(cycle_dir, "place_frame.jpg"), data["place_frame"])

    frame_info = []
    for i, f in enumerate(frames):
        fname = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, fname), f["image"])
        frame_info.append({
            "frame": fname,
            "timestamp": f["timestamp"],
            "robot_pos": f["robot_pos"],
        })

    json_data = {}
    for k, v in data.items():
        if k.endswith("_frame"):
            continue
        if isinstance(v, np.ndarray):
            json_data[k] = v.tolist()
        else:
            json_data[k] = v
    json_data["frames"] = frame_info
    json_data["total_frames"] = len(frames)

    with open(os.path.join(cycle_dir, "data.json"), "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"  데이터 저장: {cycle_dir} ({len(frames)}프레임)")


# === 변환 행렬 ===
T = compute_transform(camera_points, robot_points)
print("=== Pick and Place 자동화 ===")
avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"캘리브레이션 평균 오차: {avg_err:.1f}mm (포인트 {len(camera_points)}개)")
print(f"물체 구분: z >= {CAN_Z_THRESHOLD}mm → 캔(그리퍼 {GRIP_CAN}), z < {CAN_Z_THRESHOLD}mm → 블럭(그리퍼 {GRIP_BLOCK})")

os.makedirs(DATA_DIR, exist_ok=True)
print(f"데이터 저장: {DATA_DIR}\n")

# === 카메라 ===
clicked_point = None
click_ready = False


def mouse_callback(event, x, y, flags, param):
    global clicked_point, click_ready
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        click_ready = True


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

window_name = "Pick and Place Auto"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

print("=== 조작법 ===")
print("  1. 물체 클릭 → 자동 구분(캔/블럭) → 이동 → 자동 잡기")
print("  2. 놓을 위치 클릭 → 이동 → 놓기 → 홈 복귀")
print("  q: 종료 | r: 리셋\n")

last_frame_time = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data()).copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 연속 프레임 저장
        now = time.time()
        if recording and (now - last_frame_time) >= FRAME_INTERVAL:
            save_frame(color_image, datetime.now().isoformat())
            last_frame_time = now

        # 클릭 처리
        if clicked_point and click_ready:
            px, py = clicked_point
            dist = depth_frame.get_distance(px, py)

            if dist > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cam_xyz = np.array([point_3d[0]*1000, point_3d[1]*1000, point_3d[2]*1000])
                robot_xyz = cam_to_robot(cam_xyz, T)
                reachable, reach_dist = check_reachable(robot_xyz[0], robot_xyz[1])

                if current_state == STATE_PICK:
                    # === PICK ===
                    object_type, grip_value = classify_object(robot_xyz[2])

                    print(f"\n[PICK] {object_type} 감지!")
                    print(f"  카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")
                    print(f"  로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}")
                    print(f"  종류: {object_type} → 그리퍼: {grip_value}")

                    if not reachable:
                        print(f"  도달 불가! 거리={reach_dist:.0f}mm")
                    else:
                        if robot_xyz[2] < MIN_Z:
                            robot_xyz[2] = MIN_Z
                        pick_z = robot_xyz[2]

                        # 녹화 시작
                        recording = True
                        frame_list.clear()
                        last_frame_time = time.time()
                        save_frame(color_image, datetime.now().isoformat())

                        cycle_data = {
                            "cycle": cycle_count,
                            "object_type": object_type,
                            "grip_value": grip_value,
                            "pick_timestamp": datetime.now().isoformat(),
                            "pick_pixel": [px, py],
                            "pick_cam_xyz": cam_xyz.copy(),
                            "pick_robot_xyz": robot_xyz.copy(),
                            "pick_frame": color_image.copy(),
                        }

                        # 이동
                        move_robot(robot_xyz[0], robot_xyz[1], robot_xyz[2])

                        # 자동 잡기
                        print(f"  그리퍼 잡기 ({grip_value})...")
                        gripper_control(grip_value)
                        gripper_control(grip_value)

                        current_state = STATE_PLACE
                        print("\n  → 놓을 위치를 클릭하세요!")

                    cv2.circle(color_image, (px, py), 10, (0, 255, 0), 2)

                elif current_state == STATE_PLACE:
                    # === PLACE ===
                    print(f"\n[PLACE] 놓을 위치")
                    print(f"  카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")

                    if not reachable:
                        print(f"  도달 불가! 거리={reach_dist:.0f}mm")
                    else:
                        place_surface_z = robot_xyz[2]
                        place_z = place_surface_z + (pick_z - MIN_Z) if place_surface_z > MIN_Z else pick_z
                        if place_z < MIN_Z:
                            place_z = MIN_Z

                        print(f"  로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={place_z:.1f}")

                        cycle_data["place_timestamp"] = datetime.now().isoformat()
                        cycle_data["place_pixel"] = [px, py]
                        cycle_data["place_cam_xyz"] = cam_xyz.copy()
                        cycle_data["place_robot_xyz"] = np.array([robot_xyz[0], robot_xyz[1], place_z])
                        cycle_data["place_frame"] = color_image.copy()

                        # 이동
                        move_robot(robot_xyz[0], robot_xyz[1], place_z)

                        # 놓기
                        print("  그리퍼 열기...")
                        gripper_control(0)
                        gripper_control(0)

                        # 녹화 종료 & 저장
                        recording = False
                        save_frame(color_image, datetime.now().isoformat())
                        save_cycle(cycle_count, cycle_data, frame_list)
                        cycle_count += 1

                        # 홈 복귀
                        go_home()

                        current_state = STATE_PICK
                        print(f"\n=== 사이클 {cycle_count} 완료! 다음 물체를 클릭하세요 ===")

                    cv2.circle(color_image, (px, py), 10, (0, 0, 255), 2)
            else:
                print(f"depth 없음 ({px}, {py})")

            click_ready = False

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = False
            current_state = STATE_PICK
            gripper_control(0)
            gripper_control(0)
            print("\n=== 리셋! 물체를 클릭하세요 ===")

        # 상태 표시
        state_colors = {STATE_PICK: (0, 255, 0), STATE_PLACE: (0, 0, 255)}
        state_text = {
            STATE_PICK: f"PICK: Click object (cycle {cycle_count})",
            STATE_PLACE: f"PLACE: Click destination ({object_type} grip={grip_value})"
        }

        color = state_colors.get(current_state, (255, 255, 255))
        text = state_text.get(current_state, "")
        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if recording:
            cv2.circle(color_image, (20, 65), 10, (0, 0, 255), -1)
            cv2.putText(color_image, f"REC ({len(frame_list)} frames)", (40, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(color_image, "q=quit | r=reset", (10, 710),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, color_image)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n총 {cycle_count}개 사이클 완료. 데이터: {DATA_DIR}")
