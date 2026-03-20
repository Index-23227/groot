"""
카메라 ↔ 로봇 외부 파라미터 (T_cam2base) 캘리브레이션
ArUco 마커 기반 Hand-Eye Calibration

───────────────────────────────────────────────────────
왜 필요한가?
───────────────────────────────────────────────────────
Gemini ER이 주는 bbox_norm / grasp_point_norm 은 이미지 픽셀 좌표.
  grasp_point_norm = [0.45, 0.50]  → 픽셀 (288, 240) 정도

RealSense D435 depth 카메라로 해당 픽셀의 깊이(z)를 알면
→ 카메라 좌표계 3D 점 P_cam = (X, Y, Z)를 구할 수 있다.

그런데 로봇 팔은 로봇 베이스 좌표계(P_base)로 움직인다.
  P_base = T_cam2base @ P_cam

T_cam2base 를 모르면 "Gemini가 본 위치 = 로봇이 갈 위치" 연결 불가.
이 4×4 행렬 하나를 현장에서 한 번 측정하는 것이 이 스크립트의 목적.

───────────────────────────────────────────────────────
측정 방법 (Hand-Eye Calibration)
───────────────────────────────────────────────────────
1. ArUco 마커 (ID=0, 5cm) 를 그리퍼/TCP에 붙인다
2. 로봇을 N가지 포즈로 이동 (teach pendant, 최소 6개 권장)
3. 각 포즈에서:
   - 카메라가 마커를 감지 → T_cam2marker (카메라에서 마커까지)
   - 로봇 teach pendant에서 TCP 포즈 읽기 → T_base2tcp
4. OpenCV hand-eye calibration으로 T_cam2base 계산
5. configs/T_cam2base.npy 에 저장

사용법:
  python utils/calibration.py --run              # 대화형 캘리브레이션
  python utils/calibration.py --verify           # 저장된 결과 확인
  python utils/calibration.py --test-pixel 288 240 0.45   # 변환 테스트
"""

import sys, os, argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

CALIB_PATH = Path(__file__).parent.parent / "configs" / "T_cam2base.npy"
INTRINSICS_PATH = Path(__file__).parent.parent / "configs" / "camera_intrinsics.npy"


# ──────────────────────────────────────────────────────
# 좌표 변환 유틸
# ──────────────────────────────────────────────────────

def _Rx(a):
    return np.array([[1, 0, 0],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]])

def _Ry(a):
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]])

def _Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]])

def doosan_tcp_to_matrix(pose_mm_deg):
    """두산 TCP pose [x,y,z(mm), rx,ry,rz(deg)] → 4×4 SE3 (미터 단위).
    두산은 RPY(roll=rx, pitch=ry, yaw=rz) 고정축 회전 사용.
    """
    x, y, z, rx, ry, rz = pose_mm_deg
    R = _Rz(np.deg2rad(rz)) @ _Ry(np.deg2rad(ry)) @ _Rx(np.deg2rad(rx))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]
    return T

def matrix_to_euler_deg(R):
    """회전행렬 → RPY (deg). gimbal lock 근처 부정확할 수 있음."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2( R[2,1], R[2,2]))
        ry = np.degrees(np.arctan2(-R[2,0], sy))
        rz = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        rx = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        ry = np.degrees(np.arctan2(-R[2,0], sy))
        rz = 0.0
    return np.array([rx, ry, rz])


# ──────────────────────────────────────────────────────
# ArUco 기반 Hand-Eye Calibrator
# ──────────────────────────────────────────────────────

class ArucoCalibrator:
    def __init__(self, marker_id=0, marker_size_m=0.05):
        self.marker_id = marker_id
        self.marker_size_m = marker_size_m
        self.robot_poses = []   # list of 4×4 T_base2tcp
        self.cam_poses   = []   # list of 4×4 T_cam2marker
        self.aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.det_params  = cv2.aruco.DetectorParameters()

    def detect_marker(self, rgb_frame, camera_matrix, dist_coeffs):
        """ArUco 마커 감지 → T_cam2marker (4×4). 감지 실패 시 None."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.det_params)
        if ids is None or self.marker_id not in ids.flatten():
            return None, rgb_frame

        idx = list(ids.flatten()).index(self.marker_id)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], self.marker_size_m, camera_matrix, dist_coeffs)

        R, _ = cv2.Rodrigues(rvecs[0][0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = tvecs[0][0]

        # 시각화 (확인용)
        vis = rgb_frame.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                          rvecs[0], tvecs[0], self.marker_size_m * 0.5)
        return T, vis

    def add_sample(self, T_cam2marker, tcp_pose_mm_deg):
        T_base2tcp = doosan_tcp_to_matrix(tcp_pose_mm_deg)
        self.robot_poses.append(T_base2tcp)
        self.cam_poses.append(T_cam2marker)
        n = len(self.robot_poses)
        print(f"  [+] 샘플 {n} 추가 — TCP {tcp_pose_mm_deg[:3]} mm")

    def solve(self):
        """OpenCV TSAI 방법으로 T_cam2base 계산 (최소 4개 샘플 필요)."""
        assert len(self.robot_poses) >= 4, "최소 4개 샘플 필요"
        R_g2b = [T[:3, :3] for T in self.robot_poses]
        t_g2b = [T[:3, 3 ].reshape(3,1) for T in self.robot_poses]
        R_t2c = [T[:3, :3] for T in self.cam_poses]
        t_t2c = [T[:3, 3 ].reshape(3,1) for T in self.cam_poses]

        R_c2b, t_c2b = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        T = np.eye(4)
        T[:3, :3] = R_c2b
        T[:3, 3]  = t_c2b.flatten()
        return T

    def save(self, T_cam2base, camera_matrix, dist_coeffs):
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(CALIB_PATH), T_cam2base)
        np.save(str(INTRINSICS_PATH), {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
        })
        print(f"\n[Calib] T_cam2base → {CALIB_PATH}")
        print(f"  Translation: {T_cam2base[:3,3]*1000} mm")
        euler = matrix_to_euler_deg(T_cam2base[:3,:3])
        print(f"  Rotation (RPY deg): {euler}")
        print(f"[Calib] intrinsics → {INTRINSICS_PATH}")


# ──────────────────────────────────────────────────────
# 로드 / 검증
# ──────────────────────────────────────────────────────

def load_calibration():
    """T_cam2base (4×4), camera_matrix (3×3), dist_coeffs 로드."""
    if not CALIB_PATH.exists():
        raise FileNotFoundError(
            f"캘리브레이션 파일 없음: {CALIB_PATH}\n"
            "먼저 python utils/calibration.py --run 실행")
    T = np.load(str(CALIB_PATH))

    intrinsics = {}
    if INTRINSICS_PATH.exists():
        d = np.load(str(INTRINSICS_PATH), allow_pickle=True).item()
        intrinsics = d
    return T, intrinsics


def pixel_depth_to_robot(u, v, depth_m, T_cam2base, camera_matrix):
    """
    픽셀 (u,v) + 깊이(m) → 로봇 베이스 좌표 (mm).

    왜 이 변환이 필요한가?
    - Gemini가 grasp_point_norm = [cx, cy] 로 물체 위치를 줌
    - u = cx * W,  v = cy * H  → 픽셀
    - RealSense depth[v, u] → depth_m (물체까지 거리)
    - 핀홀 카메라 역투영으로 3D 카메라 좌표 계산
    - T_cam2base 곱해서 로봇 베이스 좌표로 변환
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    X_cam = (u - cx) * depth_m / fx
    Y_cam = (v - cy) * depth_m / fy
    Z_cam = depth_m

    P_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
    P_base = T_cam2base @ P_cam
    return P_base[:3] * 1000.0  # m → mm


# ──────────────────────────────────────────────────────
# 대화형 실행
# ──────────────────────────────────────────────────────

def run_calibration():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
                      rs.format.rgb8, CAMERA_FPS)
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()
    intr = (profile.get_stream(rs.stream.color)
                   .as_video_stream_profile().get_intrinsics())
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]])
    dist_coeffs = np.array(intr.coeffs)
    print(f"[Camera] fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    calibrator = ArucoCalibrator(marker_id=0, marker_size_m=0.05)

    print("\n=== Hand-Eye Calibration ===")
    print("준비: ArUco ID=0 (5cm) 마커를 TCP/그리퍼 끝에 붙이세요")
    print("조작: 로봇을 다양한 방향/자세로 이동 → Enter 입력 → TCP 좌표 입력")
    print("완료: 'done' 입력 (최소 6개 권장)\n")

    while True:
        n = len(calibrator.robot_poses)
        cmd = input(f"[샘플 {n}개] Enter=캡처 / done=완료 / skip=건너뜀: ").strip()

        if cmd.lower() == "done":
            if n < 4:
                print("  최소 4개 샘플 필요")
                continue
            break
        if cmd.lower() == "skip":
            continue

        # 카메라 캡처
        frames = pipeline.wait_for_frames()
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        T_cam2marker, vis = calibrator.detect_marker(rgb, camera_matrix, dist_coeffs)

        if T_cam2marker is None:
            print("  ❌ 마커 감지 실패. 카메라 뷰에 마커가 보이는지 확인.")
            # 이미지 저장해서 확인 가능하게
            import cv2 as cv
            cv.imwrite("/tmp/calib_failed.jpg", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
            print("  → /tmp/calib_failed.jpg 저장됨")
            continue

        print(f"  ✓ 마커 감지 (거리 {T_cam2marker[2,3]*100:.1f}cm)")
        tcp_str = input("  teach pendant TCP pose [x,y,z,rx,ry,rz] (mm,deg): ").strip()
        try:
            tcp = [float(v) for v in tcp_str.replace(" ", "").split(",")]
            assert len(tcp) == 6
        except Exception:
            print("  입력 형식 오류. 예: 350.5,12.3,200.0,180.0,0.0,45.0")
            continue

        calibrator.add_sample(T_cam2marker, tcp)

        # 중간 결과 (3개 이상이면)
        if len(calibrator.robot_poses) >= 3:
            try:
                T_tmp = calibrator.solve()
                print(f"  [현재 추정] 카메라 위치: {T_tmp[:3,3]*1000} mm")
            except Exception:
                pass

    T_cam2base = calibrator.solve()
    calibrator.save(T_cam2base, camera_matrix, dist_coeffs)
    pipeline.stop()
    print("\n✅ 캘리브레이션 완료")
    return T_cam2base


def verify_calibration():
    T, intr = load_calibration()
    print("T_cam2base (4×4):")
    print(np.round(T, 4))
    print(f"\n카메라 위치 (로봇 베이스 기준): {T[:3,3]*1000} mm")
    euler = matrix_to_euler_deg(T[:3, :3])
    print(f"카메라 방향 (RPY deg): {euler}")
    if "camera_matrix" in intr:
        K = intr["camera_matrix"]
        print(f"\n카메라 내부 파라미터:")
        print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f}  cx={K[0,2]:.1f} cy={K[1,2]:.1f}")


def test_pixel_conversion(u, v, depth_m):
    T, intr = load_calibration()
    K = intr.get("camera_matrix")
    if K is None:
        print("camera_matrix 없음. 캘리브레이션 재실행 필요.")
        return
    pos_mm = pixel_depth_to_robot(u, v, depth_m, T, K)
    print(f"픽셀 ({u}, {v}), 깊이 {depth_m*100:.1f}cm"
          f" → 로봇 베이스 {pos_mm} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hand-Eye Calibration (T_cam2base)")
    p.add_argument("--run",    action="store_true", help="대화형 캘리브레이션 실행")
    p.add_argument("--verify", action="store_true", help="저장된 결과 확인")
    p.add_argument("--test-pixel", nargs=3, type=float,
                   metavar=("U", "V", "DEPTH_M"),
                   help="픽셀→로봇 변환 테스트 (예: 288 240 0.45)")
    args = p.parse_args()

    if args.run:
        run_calibration()
    elif args.verify:
        verify_calibration()
    elif args.test_pixel:
        test_pixel_conversion(*args.test_pixel)
    else:
        p.print_help()
