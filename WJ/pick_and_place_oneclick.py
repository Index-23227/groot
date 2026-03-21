#!/usr/bin/env python3
"""
Pick and Place 원클릭 자동화
1. 첫번째 클릭: Pick 위치
2. 두번째 클릭: Place 위치
3. 자동으로: Pick 이동 → 잡기 → Place 이동 → 놓기 → 홈 복귀
4. 반복
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess
import os
import time

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

# === 상태 ===
STATE_CLICK_PICK = 0    # Pick 위치 클릭 대기
STATE_CLICK_PLACE = 1   # Place 위치 클릭 대기
STATE_RUNNING = 2       # 실행 중

current_state = STATE_CLICK_PICK
current_pos = [453.0, 0.0, 400.0]
cycle_count = 0

# Pick/Place 저장
pick_info = None
place_info = None
pick_marker = None
place_marker = None


def compute_transform(cam_pts, rob_pts):
    n = cam_pts.shape[0]
    A = np.hstack([cam_pts, np.ones((n, 1))])
    T, _, _, _ = np.linalg.lstsq(A, rob_pts, rcond=None)
    return T


def cam_to_robot(cam_xyz, T):
    cam_h = np.append(cam_xyz, 1.0)
    return cam_h @ T


def move_line(x, y, z):
    """move_line 명령 전송 - rx/ry/rz 절대 고정"""
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
    """z상승 → xy이동 → z하강 완전 분리"""
    if z < MIN_Z:
        z = MIN_Z

    # 1단계: z축만 안전높이로 상승 (현재 xy 유지)
    print(f"  ↑ z축 상승 (z={SAFE_Z}mm)")
    move_line(current_pos[0], current_pos[1], SAFE_Z)
    current_pos[2] = SAFE_Z

    # 2단계: xy만 목표로 이동 (z=SAFE_Z 유지)
    print(f"  → xy 이동 (x={x:.1f}, y={y:.1f})")
    move_line(x, y, SAFE_Z)
    current_pos[0] = x
    current_pos[1] = y

    # 3단계: z축만 목표높이로 하강 (xy 유지)
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


def check_reachable(x, y):
    dist = np.sqrt(x**2 + y**2)
    return dist <= MAX_REACH, dist


def classify_object(robot_z):
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


def run_pick_and_place(pick, place):
    """Pick and Place 한 사이클 실행"""
    obj_type, grip_val = classify_object(pick["robot_z"])

    print(f"\n{'='*50}")
    print(f"  사이클 {cycle_count + 1} 시작! ({obj_type})")
    print(f"{'='*50}")

    # 1. Pick 위치로 이동
    print(f"\n[1/5] Pick 이동 → x={pick['robot_x']:.1f}, y={pick['robot_y']:.1f}, z={pick['robot_z']:.1f}")
    move_robot(pick["robot_x"], pick["robot_y"], pick["robot_z"])

    # 2. 잡기
    print(f"\n[2/5] 그리퍼 잡기 ({grip_val})")
    gripper_control(grip_val)
    time.sleep(1)
    gripper_control(grip_val)
    time.sleep(1)

    # 3. Place 위치로 이동
    pick_z = pick["robot_z"]
    place_surface_z = place["robot_z"]
    place_z = place_surface_z + (pick_z - MIN_Z) if place_surface_z > MIN_Z else pick_z
    place_z += 20  # place 오프셋 +20
    if obj_type == "블럭":
        place_z += 10  # 블럭은 추가 +10
    if place_z < MIN_Z:
        place_z = MIN_Z

    print(f"\n[3/5] Place 이동 → x={place['robot_x']:.1f}, y={place['robot_y']:.1f}, z={place_z:.1f}")
    move_robot(place["robot_x"], place["robot_y"], place_z)

    # 4. 놓기
    print(f"\n[4/5] 그리퍼 열기")
    gripper_control(0)
    time.sleep(1)
    gripper_control(0)
    time.sleep(1)

    # 5. 홈 복귀
    print(f"\n[5/5] 홈 복귀")
    go_home()

    print(f"\n{'='*50}")
    print(f"  사이클 완료!")
    print(f"{'='*50}\n")


# === 변환 행렬 ===
T = compute_transform(camera_points, robot_points)
print("=== Pick and Place 원클릭 ===")
avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"캘리브레이션 평균 오차: {avg_err:.1f}mm ({len(camera_points)}포인트)")
print(f"캔: z >= {CAN_Z_THRESHOLD}mm → 그리퍼 {GRIP_CAN}")
print(f"블럭: z < {CAN_Z_THRESHOLD}mm → 그리퍼 {GRIP_BLOCK}\n")

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

window_name = "Pick and Place OneClick"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

print("=== 조작법 ===")
print("  첫번째 클릭: Pick 위치 (물체)")
print("  두번째 클릭: Place 위치 (놓을 곳)")
print("  → 자동으로 Pick → 잡기 → Place → 놓기 → 홈 복귀")
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

        # Pick 마커 표시
        if pick_marker:
            cv2.circle(color_image, pick_marker, 12, (0, 255, 0), 3)
            cv2.putText(color_image, "PICK", (pick_marker[0]+15, pick_marker[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 클릭 처리
        if clicked_point and click_ready and current_state != STATE_RUNNING:
            px, py = clicked_point
            dist = depth_frame.get_distance(px, py)

            if dist > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cam_xyz = np.array([point_3d[0]*1000, point_3d[1]*1000, point_3d[2]*1000])
                robot_xyz = cam_to_robot(cam_xyz, T)
                reachable, reach_dist = check_reachable(robot_xyz[0], robot_xyz[1])

                if not reachable:
                    print(f"  도달 불가! 거리={reach_dist:.0f}mm")
                elif current_state == STATE_CLICK_PICK:
                    # Pick 위치 저장
                    if robot_xyz[2] < MIN_Z:
                        robot_xyz[2] = MIN_Z
                    obj_type, grip_val = classify_object(robot_xyz[2])

                    pick_info = {
                        "robot_x": robot_xyz[0],
                        "robot_y": robot_xyz[1],
                        "robot_z": robot_xyz[2],
                    }
                    pick_marker = (px, py)

                    print(f"\n[PICK 설정] {obj_type}")
                    print(f"  카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")
                    print(f"  로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}")
                    print(f"  → 이제 Place 위치를 클릭하세요!")

                    current_state = STATE_CLICK_PLACE

                elif current_state == STATE_CLICK_PLACE:
                    # Place 위치 저장 & 실행
                    place_info = {
                        "robot_x": robot_xyz[0],
                        "robot_y": robot_xyz[1],
                        "robot_z": robot_xyz[2],
                    }
                    place_marker = (px, py)

                    print(f"\n[PLACE 설정]")
                    print(f"  카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")
                    print(f"  로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}")

                    # 실행!
                    current_state = STATE_RUNNING
                    cv2.circle(color_image, (px, py), 12, (0, 0, 255), 3)
                    cv2.putText(color_image, "PLACE", (px+15, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow(window_name, color_image)
                    cv2.waitKey(1)

                    run_pick_and_place(pick_info, place_info)

                    cycle_count += 1
                    pick_marker = None
                    place_marker = None
                    current_state = STATE_CLICK_PICK
                    print("다음 Pick 위치를 클릭하세요!\n")
            else:
                print(f"depth 없음 ({px}, {py})")

            click_ready = False

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_state = STATE_CLICK_PICK
            pick_marker = None
            place_marker = None
            gripper_control(0)
            gripper_control(0)
            print("\n=== 리셋! Pick 위치를 클릭하세요 ===")

        # 상태 표시
        if current_state == STATE_CLICK_PICK:
            text = f"1. Pick 위치 클릭 (cycle {cycle_count})"
            color = (0, 255, 0)
        elif current_state == STATE_CLICK_PLACE:
            text = "2. Place 위치 클릭"
            color = (0, 0, 255)
        else:
            text = "실행 중..."
            color = (0, 255, 255)

        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(color_image, "q=quit | r=reset", (10, 710),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, color_image)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n총 {cycle_count}개 사이클 완료")
