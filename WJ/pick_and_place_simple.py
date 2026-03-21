#!/usr/bin/env python3
"""
Pick and Place - 카메라 클릭으로 로봇 제어 (데이터 저장 없음)
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# === 캘리브레이션 데이터 ===
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
], dtype=np.float64)

# === 설정 ===
SAFE_Z = 400
MIN_Z = 120
MAX_REACH = 800

# === 상태 ===
STATE_PICK = "PICK"
STATE_GRIP = "GRIP"
STATE_PLACE = "PLACE"

current_state = STATE_PICK
current_pos = [453.0, 0.0, 300.0]
pick_z = 0
grip_value = 0


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


def move_z_only(z, rz=93.4):
    """z축만 이동 (xy 완전 고정)"""
    if z < MIN_Z:
        z = MIN_Z
    move_line(current_pos[0], current_pos[1], z, rz)
    current_pos[2] = z


def move_xy_only(x, y, rz=93.4):
    """xy만 이동 (z 완전 고정)"""
    move_line(x, y, current_pos[2], rz)
    current_pos[0] = x
    current_pos[1] = y


def move_robot(x, y, z):
    if z < MIN_Z:
        print(f"  안전 제한: z={z:.1f} → {MIN_Z}mm")
        z = MIN_Z

    # 1단계: z축만 올라가기
    print(f"  ↑ 상승 (z={SAFE_Z}mm)")
    move_z_only(SAFE_Z)

    # 2단계: xy만 이동
    print(f"  → XY 이동 (x={x:.1f}, y={y:.1f})")
    move_xy_only(x, y)

    # 3단계: z축만 내려가기
    print(f"  ↓ 하강 (z={z:.1f}mm)")
    move_z_only(z)

    print("이동 완료!")


def gripper_control(position):
    position = max(0, min(700, position))
    cmd = (
        f'ros2 topic pub /dsr01/gripper/position_cmd std_msgs/msg/Int32 '
        f'"{{data: {position}}}" --once'
    )
    print(f"  그리퍼: {position}")
    subprocess.run(cmd, shell=True, env=os.environ.copy(), capture_output=True)


def check_reachable(x, y, z):
    dist = np.sqrt(x**2 + y**2)
    return dist <= MAX_REACH, dist


# === 변환 행렬 ===
T = compute_transform(camera_points, robot_points)
print("=== Pick and Place ===")
avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"캘리브레이션 평균 오차: {avg_err:.1f}mm\n")

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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

window_name = "Pick and Place"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

print("=== 조작법 ===")
print("  1. 물체를 클릭 → 그리퍼가 이동")
print("  2. 터미널에서 그리퍼 값 입력 (0~700) → 잡기")
print("  3. 놓을 위치 클릭 → 이동 후 자동으로 놓기")
print("  4. 반복!")
print("  q: 종료 | r: 리셋\n")

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

        if clicked_point and click_ready:
            px, py = clicked_point
            dist = depth_frame.get_distance(px, py)

            if dist > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cam_xyz = np.array([point_3d[0]*1000, point_3d[1]*1000, point_3d[2]*1000])
                robot_xyz = cam_to_robot(cam_xyz, T)
                reachable, reach_dist = check_reachable(robot_xyz[0], robot_xyz[1], robot_xyz[2])

                if current_state == STATE_PICK:
                    print(f"\n[PICK] 물체 위치")
                    print(f"  카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")
                    print(f"  로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}")

                    if not reachable:
                        print(f"  도달 불가! 거리={reach_dist:.0f}mm")
                    else:
                        if robot_xyz[2] < MIN_Z:
                            robot_xyz[2] = MIN_Z
                        pick_z = robot_xyz[2]
                        move_robot(robot_xyz[0], robot_xyz[1], robot_xyz[2])
                        current_state = STATE_GRIP
                        print("\n  → 그리퍼 값을 입력하세요 (0~700): ", end="", flush=True)

                    cv2.circle(color_image, (px, py), 10, (0, 255, 0), 2)

                elif current_state == STATE_PLACE:
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

                        move_robot(robot_xyz[0], robot_xyz[1], place_z)

                        print("  그리퍼 열기...")
                        gripper_control(0)
                        gripper_control(0)

                        print("  홈 위치로 복귀...")
                        home_cmd = (
                            'ros2 service call /dsr01/motion/move_joint dsr_msgs2/srv/MoveJoint '
                            '"{pos: [0.0, 0.0, 90.0, 0.0, 90.0, 90.0], vel: 30.0, acc: 30.0}"'
                        )
                        subprocess.run(home_cmd, shell=True, env=os.environ.copy(), capture_output=True)
                        current_pos[0] = 453.0
                        current_pos[1] = 0.0
                        current_pos[2] = 400.0

                        current_state = STATE_PICK
                        print("\n=== 완료! 다음 물체를 클릭하세요 ===")

                    cv2.circle(color_image, (px, py), 10, (0, 0, 255), 2)
            else:
                print(f"depth 없음 ({px}, {py})")

            click_ready = False

        if current_state == STATE_GRIP:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            import select
            import sys
            if select.select([sys.stdin], [], [], 0)[0]:
                val = sys.stdin.readline().strip()
                if val:
                    try:
                        grip_value = max(0, min(700, int(val)))
                        gripper_control(grip_value)
                        gripper_control(grip_value)
                        current_state = STATE_PLACE
                        print("\n  → 놓을 위치를 클릭하세요!")
                    except ValueError:
                        print("  숫자를 입력하세요: ", end="", flush=True)
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_state = STATE_PICK
                gripper_control(0)
                print("\n=== 리셋! 물체를 클릭하세요 ===")

        state_colors = {STATE_PICK: (0, 255, 0), STATE_GRIP: (0, 255, 255),
                       STATE_PLACE: (0, 0, 255)}
        state_text = {STATE_PICK: "PICK: Click object",
                     STATE_GRIP: "GRIP: Enter value in terminal",
                     STATE_PLACE: "PLACE: Click destination"}

        color = state_colors.get(current_state, (255, 255, 255))
        text = state_text.get(current_state, "")
        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(color_image, "q=quit | r=reset", (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow(window_name, color_image)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n종료")
