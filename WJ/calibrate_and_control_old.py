#!/usr/bin/env python3
"""
리얼센스 카메라 클릭 → 로봇 그리퍼 자동 이동
카메라 좌표 → 로봇 좌표 변환 (캘리브레이션 데이터 기반)
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import subprocess
import json

# === 캘리브레이션 데이터 (수집된 포인트) ===
# [카메라 x, y, z] → [로봇 x, y, z]
camera_points = np.array([
    [-2.8, -127.1, 685.0],
    [102.4, -47.4, 636.0],
    [-91.5, 53.8, 576.0],
    [17.5, 104.4, 545.0],
    [-209.6, -172.4, 731.0],
    [-194.9, -128.4, 741.0],
    [6.1, -57.1, 695.0],
    [159.2, -67.2, 705.0],
    [-144.3, 35.7, 653.0],
    [215.8, 68.7, 644.0],
], dtype=np.float64)

robot_points = np.array([
    [453.0, 0.0, 190.0],
    [553.0, 100.0, 190.0],
    [653.0, -100.0, 190.0],
    [720.0, 0.0, 190.0],
    [400.0, -200.0, 190.0],
    [400.0, -200.0, 150.0],
    [500.0, 0.0, 150.0],
    [500.0, 150.0, 150.0],
    [600.0, -150.0, 150.0],
    [650.0, 200.0, 150.0],
], dtype=np.float64)


def compute_transform(cam_pts, rob_pts):
    """카메라 → 로봇 변환 행렬 계산 (최소자승법)"""
    n = cam_pts.shape[0]

    # 카메라 좌표에 1 추가 [x, y, z, 1]
    A = np.hstack([cam_pts, np.ones((n, 1))])

    # 각 축별로 최소자승법
    # robot = A @ T
    T, residuals, rank, sv = np.linalg.lstsq(A, rob_pts, rcond=None)

    return T


def cam_to_robot(cam_xyz, T):
    """카메라 좌표 → 로봇 좌표 변환"""
    cam_h = np.append(cam_xyz, 1.0)
    robot_xyz = cam_h @ T
    return robot_xyz


SAFE_Z = 300  # 이동 시 안전 높이 (mm)
MIN_Z = 120   # 최소 z값 (mm)
MAX_REACH = 800  # 로봇 최대 도달 거리 (mm)

def check_reachable(x, y, z):
    """로봇 도달 가능 여부 확인"""
    dist = np.sqrt(x**2 + y**2)
    if dist > MAX_REACH:
        return False, dist
    return True, dist


def move_line(x, y, z, rz=93.4):
    """로봇을 xyz 좌표로 직선 이동"""
    if z < MIN_Z:
        print(f"안전 제한: z={z:.1f}mm → {MIN_Z}mm")
        z = MIN_Z
    cmd = (
        f'ros2 service call /dsr01/motion/move_line dsr_msgs2/srv/MoveLine '
        f'"{{pos: [{x:.1f}, {y:.1f}, {z:.1f}, 3.4, -180.0, {rz}], '
        f'vel: [50.0, 50.0], acc: [50.0, 50.0]}}"'
    )
    subprocess.run(cmd, shell=True,
                   env={**__import__('os').environ},
                   capture_output=True)


def move_robot(x, y, z, rz=93.4):
    """로봇을 xyz로 이동 (위로 올라가고 → XY 이동 → 내려가기)"""
    if z < MIN_Z:
        print(f"안전 제한: z={z:.1f}mm → {MIN_Z}mm")
        z = MIN_Z

    # 1. 현재 위치에서 안전 높이로 올라가기
    print(f"  ↑ 안전 높이({SAFE_Z}mm)로 상승")
    move_line(current_pos[0], current_pos[1], SAFE_Z, rz)

    # 2. 안전 높이에서 목표 XY로 이동
    print(f"  → XY 이동: x={x:.1f}, y={y:.1f}")
    move_line(x, y, SAFE_Z, rz)

    # 3. 목표 Z로 하강
    print(f"  ↓ 목표 높이({z:.1f}mm)로 하강")
    move_line(x, y, z, rz)

    # 현재 위치 업데이트
    current_pos[0] = x
    current_pos[1] = y
    current_pos[2] = z
    print("이동 완료!")


def gripper_control(position):
    """그리퍼 제어 (0=열림, 700=닫힘)"""
    cmd = (
        f'ros2 topic pub /dsr01/gripper/position_cmd std_msgs/msg/Int32 '
        f'"{{data: {position}}}" --once'
    )
    print(f"그리퍼: {position} ({'닫기' if position > 0 else '열기'})")
    subprocess.run(cmd, shell=True,
                   env={**__import__('os').environ},
                   capture_output=True)


# 현재 위치 추적 [x, y, z]
current_pos = [453.0, 0.0, 200.0]


# === 변환 행렬 계산 ===
T = compute_transform(camera_points, robot_points)
print("=== 캘리브레이션 완료 ===")
print(f"변환 행렬:\n{T}")

# 정확도 검증
print("\n=== 검증 ===")
for i in range(len(camera_points)):
    pred = cam_to_robot(camera_points[i], T)
    actual = robot_points[i]
    err = np.linalg.norm(pred - actual)
    print(f"  포인트 {i+1}: 예측={pred.astype(int)}, 실제={actual.astype(int)}, 오차={err:.1f}mm")

avg_err = np.mean([np.linalg.norm(cam_to_robot(camera_points[i], T) - robot_points[i])
                    for i in range(len(camera_points))])
print(f"\n평균 오차: {avg_err:.1f}mm")

# === 리얼센스 카메라 시작 ===
clicked_point = None
move_requested = False

def mouse_callback(event, x, y, flags, param):
    global clicked_point, move_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        move_requested = True

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

cv2.namedWindow("Click to Move Robot")
cv2.setMouseCallback("Click to Move Robot", mouse_callback)

last_cam = None
last_robot = None

print("\n=== 카메라 화면에서 클릭하면 로봇이 이동합니다 ===")
print("  왼쪽 클릭: 해당 위치로 로봇 이동")
print("  g: 그리퍼 닫기 (500)")
print("  o: 그리퍼 열기 (0)")
print("  q: 종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        if clicked_point and move_requested:
            px, py = clicked_point
            dist = depth_frame.get_distance(px, py)
            if dist > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cam_xyz = np.array([point_3d[0]*1000, point_3d[1]*1000, point_3d[2]*1000])
                robot_xyz = cam_to_robot(cam_xyz, T)

                last_cam = cam_xyz
                last_robot = robot_xyz

                print(f"\n카메라: x={cam_xyz[0]:.1f}, y={cam_xyz[1]:.1f}, z={cam_xyz[2]:.1f}")
                print(f"로봇:   x={robot_xyz[0]:.1f}, y={robot_xyz[1]:.1f}, z={robot_xyz[2]:.1f}")

                # 안전 체크: z는 절대 120mm 이하로 내려가지 않음
                if robot_xyz[2] < MIN_Z:
                    print(f"경고: z={robot_xyz[2]:.1f}mm → {MIN_Z}mm로 제한")
                    robot_xyz[2] = MIN_Z

                # 도달 가능 여부 확인
                reachable, dist = check_reachable(robot_xyz[0], robot_xyz[1], robot_xyz[2])
                if not reachable:
                    print(f"경고: 도달 불가! 거리={dist:.0f}mm > {MAX_REACH}mm. 이동 취소.")
                else:
                    move_robot(robot_xyz[0], robot_xyz[1], robot_xyz[2])
                    val = input("그리퍼 값 입력 (0~700, Enter=스킵): ").strip()
                    if val:
                        try:
                            gripper_control(max(0, min(700, int(val))))
                        except ValueError:
                            print("스킵")

                # 클릭 위치에 원 표시
                cv2.circle(color_image, (px, py), 10, (0, 255, 0), 2)

            move_requested = False

        # 화면에 정보 표시
        if last_cam is not None:
            cv2.putText(color_image, f"Cam: {last_cam[0]:.0f}, {last_cam[1]:.0f}, {last_cam[2]:.0f}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, f"Robot: {last_robot[0]:.0f}, {last_robot[1]:.0f}, {last_robot[2]:.0f}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(color_image, "Click object to move robot | q=quit",
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Click to Move Robot", color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            gripper_control(0)
        elif key == ord('1'):
            gripper_control(100)
        elif key == ord('2'):
            gripper_control(200)
        elif key == ord('3'):
            gripper_control(300)
        elif key == ord('4'):
            gripper_control(400)
        elif key == ord('5'):
            gripper_control(500)
        elif key == ord('6'):
            gripper_control(600)
        elif key == ord('7'):
            gripper_control(700)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n종료")
