"""
4점 Extrinsic Calibration: 카메라 좌표계 → 로봇 베이스 좌표계

사용법 (현장):
  1. 로봇 TCP를 작업대 위 4개 점으로 이동 (teach pendant)
  2. 각 점에서 TCP 좌표(mm) + 카메라 pixel(x,y) + depth(mm) 기록
  3. 아래 main 블록의 값을 수정 후 실행
  4. T_cam2base.npy 저장됨
"""

import numpy as np

# ===== 현장에서 수정 =====
CAMERA_INTRINSICS = {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
}


def pixel_depth_to_cam3d(px, py, depth_mm, intrinsics):
    """2D pixel + depth → 카메라 프레임 3D (mm)"""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x = (px - cx) * depth_mm / fx
    y = (py - cy) * depth_mm / fy
    z = depth_mm
    return np.array([x, y, z])


def calibrate_4point(pixel_points, depth_values, robot_points, intrinsics):
    """
    4개 대응점으로 rigid transform T_cam2base 계산.

    Returns:
        T_cam2base: (4, 4) 변환 행렬
    """
    cam_points = np.array([
        pixel_depth_to_cam3d(px, py, d, intrinsics)
        for (px, py), d in zip(pixel_points, depth_values)
    ])
    robot_points = np.array(robot_points, dtype=float)

    cam_c = cam_points.mean(axis=0)
    rob_c = robot_points.mean(axis=0)

    H = (cam_points - cam_c).T @ (robot_points - rob_c)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = rob_c - R @ cam_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    print("Calibration errors:")
    for i in range(len(cam_points)):
        transformed = R @ cam_points[i] + t
        err = np.linalg.norm(transformed - robot_points[i])
        print(f"  Point {i}: {err:.1f} mm")

    return T


def pixel_depth_to_robot(px, py, depth_mm, intrinsics, T_cam2base):
    """단일 픽셀 + depth → 로봇 베이스 프레임 3D 좌표 (mm)"""
    cam_pt = pixel_depth_to_cam3d(px, py, depth_mm, intrinsics)
    cam_pt_h = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
    robot_pt = T_cam2base @ cam_pt_h
    return robot_pt[:3]


if __name__ == "__main__":
    # ===== 현장에서 아래 값을 실측값으로 교체 =====
    pixel_points = [
        (150, 120),
        (490, 120),
        (150, 360),
        (490, 360),
    ]
    depth_values = [650, 655, 648, 652]  # mm
    robot_points = [
        [300, -200, 50],
        [300, 200, 50],
        [500, -200, 50],
        [500, 200, 50],
    ]  # mm

    T = calibrate_4point(pixel_points, depth_values, robot_points, CAMERA_INTRINSICS)
    print("\nT_cam2base:")
    print(T)
    np.save("T_cam2base.npy", T)
    print("\nSaved to T_cam2base.npy")
