"""
Track 2: Gemini-ER + MoveL + RGBD — Zero-shot Pick & Place Pipeline

핵심: cuRobo/ikpy IK 대신 두산 내장 MoveL(직교좌표 이동) 사용.
      두산 컨트롤러가 IK + 경로 생성 + joint limit 검사를 내부에서 처리.

Usage:
  # 단일 태스크
  python track2/track2_main.py --instruction "파란 약병을 조제함에 넣어"

  # 복합 태스크 (Gemini가 자동 분해)
  python track2/track2_main.py --instruction "약병들을 색깔별로 분류해" --mode multi

  # 모듈 테스트
  python track2/track2_main.py --test-pointing    # Gemini pointing만 테스트
  python track2/track2_main.py --test-calib       # 캘리브레이션 검증
  python track2/track2_main.py --test-movel       # MoveL 단독 테스트
"""

import argparse
import json
import time
import cv2
import numpy as np

from calibration import pixel_depth_to_robot, CAMERA_INTRINSICS
from gemini_er_client import (
    get_pick_point, get_place_point,
    decompose_task, check_progress, replan_after_failure,
    predict_next_action_icl,
)

# ===== 현장에서 수정 =====
ROBOT_IP = "192.168.137.100"
CALIBRATION_PATH = "T_cam2base.npy"

# MoveL 기본 orientation: 그리퍼가 정확히 아래를 향함
# [rx, ry, rz] in degrees
DEFAULT_ORIENTATION = [0.0, 180.0, 0.0]

# Safety
APPROACH_HEIGHT_MM = 100    # 물체 위 10cm에서 접근
RETREAT_HEIGHT_MM = 150     # pick 후 15cm 들어올리기
PLACE_RELEASE_HEIGHT_MM = 30  # place 시 표면 위 3cm에서 release
GRIPPER_CLOSE_WAIT = 1.5   # 그리퍼 닫힘 대기 (초)
MAX_RETRIES = 3
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint deg

# MoveL 속도/가속도
VEL_FAST = [200.0, 50.0]     # [mm/s, deg/s] 이동용
VEL_SLOW = [50.0, 20.0]      # pick/place 접근용
ACC_FAST = [200.0, 50.0]
ACC_SLOW = [50.0, 20.0]


# ===== Robot Interface =====
class DoosanInterface:
    """두산 E0509 MoveL + Gripper 인터페이스"""

    def __init__(self):
        self.mock = True
        self.gripper_stroke = 0
        self.joint_positions = np.zeros(6)
        try:
            import rclpy
            from dsr_msgs2.srv import MoveJoint, MoveLine
            from std_srvs.srv import Trigger
            from std_msgs.msg import Int32
            from sensor_msgs.msg import JointState
            try:
                rclpy.init()
            except RuntimeError:
                pass
            self.node = rclpy.create_node("track2_controller")
            # MoveL (핵심 — 직교좌표 이동, 내장 IK 사용)
            self.move_line_cli = self.node.create_client(
                MoveLine, "/dsr01/motion/move_line")
            # MoveJ (home 복귀용)
            self.move_joint_cli = self.node.create_client(
                MoveJoint, "/dsr01/motion/move_joint")
            # Gripper
            self.gripper_open_cli = self.node.create_client(
                Trigger, "/dsr01/gripper/open")
            self.gripper_close_cli = self.node.create_client(
                Trigger, "/dsr01/gripper/close")
            # Subscriptions
            self.node.create_subscription(
                Int32, "/dsr01/gripper/stroke", self._stroke_cb, 10)
            self.node.create_subscription(
                JointState, "/joint_states", self._joint_cb, 10)
            self.mock = False
            print("[Robot] ROS2 connected (MoveL mode)")
        except ImportError:
            print("[Robot] MOCK mode (ROS2 not available)")

    def _stroke_cb(self, msg):
        self.gripper_stroke = msg.data

    def _joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6])

    def move_line(self, x, y, z, rx=0.0, ry=180.0, rz=0.0,
                  vel=None, acc=None):
        """
        MoveL: 직교좌표로 TCP 이동 (두산 내장 IK가 자동 해결).

        Args:
            x, y, z: mm (로봇 베이스 프레임)
            rx, ry, rz: degree (TCP orientation)
            vel: [linear_mm/s, angular_deg/s]
            acc: [linear_mm/s2, angular_deg/s2]
        """
        vel = vel or VEL_FAST
        acc = acc or ACC_FAST
        pos = [float(x), float(y), float(z),
               float(rx), float(ry), float(rz)]

        if self.mock:
            print(f"[MOCK] MoveL → [{x:.1f}, {y:.1f}, {z:.1f}, "
                  f"{rx:.0f}, {ry:.0f}, {rz:.0f}]")
            return True
        import rclpy
        from dsr_msgs2.srv import MoveLine
        req = MoveLine.Request()
        req.pos = pos
        req.vel = vel
        req.acc = acc
        future = self.move_line_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        if future.result() is not None:
            return True
        print(f"[Robot] MoveL FAILED → {pos}")
        return False

    def move_joint(self, joint_deg, vel=30.0, acc=30.0):
        """MoveJ: joint 이동 (home 복귀용)"""
        if self.mock:
            print(f"[MOCK] MoveJ → [{', '.join(f'{d:.1f}' for d in joint_deg)}]")
            return True
        import rclpy
        from dsr_msgs2.srv import MoveJoint
        req = MoveJoint.Request()
        req.pos = list(joint_deg)
        req.vel = vel
        req.acc = acc
        future = self.move_joint_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        return future.result() is not None

    def gripper_open(self):
        if self.mock:
            self.gripper_stroke = 0
            print("[MOCK] Gripper OPEN")
            return
        import rclpy
        from std_srvs.srv import Trigger
        req = Trigger.Request()
        future = self.gripper_open_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        time.sleep(1.0)
        print("[Robot] Gripper OPEN")

    def gripper_close(self):
        if self.mock:
            self.gripper_stroke = 350
            print("[MOCK] Gripper CLOSE")
            return
        import rclpy
        from std_srvs.srv import Trigger
        req = Trigger.Request()
        future = self.gripper_close_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        time.sleep(GRIPPER_CLOSE_WAIT)
        print(f"[Robot] Gripper CLOSE (stroke={self.gripper_stroke})")

    def is_object_grasped(self, min_threshold=50, max_threshold=650):
        """
        Stroke 기반 grasp 판별.
        - stroke ~ 0: 열림
        - stroke 50~650: 물체 잡힘 (중간에 멈춤)
        - stroke ~ 700: 완전 닫힘 (허공에서 닫힘 = 물체 없음)
        """
        if self.mock:
            return min_threshold < self.gripper_stroke < max_threshold
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.1)
        grasped = min_threshold < self.gripper_stroke < max_threshold
        print(f"[Robot] Grasp check: stroke={self.gripper_stroke}, "
              f"grasped={grasped}")
        return grasped

    def go_home(self):
        """Home 위치로 복귀"""
        return self.move_joint(HOME_POSITION)


# ===== Camera Interface =====
class RGBDCamera:
    """RGBD 카메라 (RealSense / 일반 USB)"""

    def __init__(self, index=0):
        self.depth_frame = None
        self.use_realsense = False
        try:
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = self.pipeline.start(cfg)
            # Intrinsics 자동 획득
            intr = profile.get_stream(
                rs.stream.color).as_video_stream_profile().get_intrinsics()
            CAMERA_INTRINSICS["fx"] = intr.fx
            CAMERA_INTRINSICS["fy"] = intr.fy
            CAMERA_INTRINSICS["cx"] = intr.ppx
            CAMERA_INTRINSICS["cy"] = intr.ppy
            self.use_realsense = True
            print(f"[Camera] RealSense (fx={intr.fx:.1f}, fy={intr.fy:.1f})")
        except (ImportError, Exception):
            self.cap = cv2.VideoCapture(index)
            print(f"[Camera] OpenCV index {index} (depth=MOCK)")

    def read(self):
        """RGB + Depth 반환"""
        if self.use_realsense:
            import pyrealsense2 as rs
            frames = self.pipeline.wait_for_frames()
            color = np.asanyarray(frames.get_color_frame().get_data())
            self.depth_frame = np.asanyarray(
                frames.get_depth_frame().get_data())
            return color, self.depth_frame
        ret, color = self.cap.read()
        return (color if ret else None), self.depth_frame

    def get_depth_at(self, x, y):
        """특정 픽셀의 depth (mm). 5x5 median으로 noise 제거."""
        if self.depth_frame is None:
            return 600.0  # MOCK: 60cm 고정
        region = self.depth_frame[max(0, y-2):y+3, max(0, x-2):x+3]
        valid = region[region > 0]
        return float(np.median(valid)) if len(valid) > 0 else None

    def release(self):
        if self.use_realsense:
            self.pipeline.stop()
        elif hasattr(self, "cap"):
            self.cap.release()


# ===== Main Pipeline =====
class Track2Pipeline:
    """Gemini-ER + MoveL + RGBD Zero-shot Pipeline"""

    def __init__(self, args):
        self.camera = RGBDCamera(args.camera_index)
        self.robot = DoosanInterface()
        try:
            self.T_cam2base = np.load(CALIBRATION_PATH)
            print(f"[Calib] Loaded {CALIBRATION_PATH}")
        except FileNotFoundError:
            print(f"[Calib] {CALIBRATION_PATH} not found — "
                  f"run calibration.py first!")
            self.T_cam2base = np.eye(4)

        print(f"\n{'='*60}")
        print(f"  Track 2: Gemini-ER + MoveL + RGBD (Zero-shot)")
        print(f"  MoveL: 두산 내장 IK (cuRobo 불필요)")
        print(f"{'='*60}\n")

    def pixel_to_robot_3d(self, px, py, depth_mm):
        """Pixel + depth → 로봇 베이스 프레임 3D 좌표 (mm)"""
        return pixel_depth_to_robot(
            px, py, depth_mm, CAMERA_INTRINSICS, self.T_cam2base)

    def move_to_xyz(self, x, y, z, vel=None, acc=None):
        """직교좌표로 이동 (MoveL, 그리퍼 아래 방향)"""
        return self.robot.move_line(
            x, y, z,
            DEFAULT_ORIENTATION[0], DEFAULT_ORIENTATION[1],
            DEFAULT_ORIENTATION[2],
            vel=vel, acc=acc)

    def pick(self, instruction):
        """단일 pick 동작: Gemini pointing → RGBD → MoveL → grasp"""
        color, _ = self.camera.read()
        if color is None:
            return False

        # 1. Gemini로 물체 위치 찾기
        print(f"\n[Gemini] Finding: {instruction}")
        point = get_pick_point(color, instruction)
        if point is None:
            return False
        px, py = point
        print(f"[Gemini] -> pixel ({px}, {py})")

        # 2. Depth 읽기
        depth = self.camera.get_depth_at(px, py)
        if depth is None:
            print("[Camera] No depth at target pixel!")
            return False

        # 3. Pixel+depth → 로봇 3D
        target = self.pixel_to_robot_3d(px, py, depth)
        print(f"[3D] -> robot [{target[0]:.1f}, {target[1]:.1f}, "
              f"{target[2]:.1f}] mm")

        # 4. Gripper 열기
        self.robot.gripper_open()

        # 5. Approach: 물체 위에서 접근 (MoveL)
        if not self.move_to_xyz(target[0], target[1],
                                target[2] + APPROACH_HEIGHT_MM,
                                vel=VEL_FAST, acc=ACC_FAST):
            return False
        time.sleep(0.3)

        # 6. Descend: 물체 높이로 내려가기 (느리게)
        if not self.move_to_xyz(target[0], target[1], target[2],
                                vel=VEL_SLOW, acc=ACC_SLOW):
            return False
        time.sleep(0.3)

        # 7. Gripper 닫기
        self.robot.gripper_close()

        # 8. Grasp 성공 확인
        if not self.robot.is_object_grasped():
            print("[Grasp] FAILED — object not in gripper")
            return False

        # 9. Retreat: 들어올리기
        self.move_to_xyz(target[0], target[1],
                         target[2] + RETREAT_HEIGHT_MM,
                         vel=VEL_SLOW, acc=ACC_SLOW)
        print("[Pick] SUCCESS")
        return True

    def place(self, instruction):
        """단일 place 동작: Gemini pointing → RGBD → MoveL → release"""
        color, _ = self.camera.read()
        if color is None:
            return False

        # 1. Gemini로 place 위치 찾기
        print(f"\n[Gemini] Place location: {instruction}")
        point = get_place_point(color, instruction)
        if point is None:
            return False
        px, py = point

        # 2. Depth + 3D 변환
        depth = self.camera.get_depth_at(px, py)
        if depth is None:
            return False
        target = self.pixel_to_robot_3d(px, py, depth)

        # 3. Approach: 위에서 접근
        if not self.move_to_xyz(target[0], target[1],
                                target[2] + APPROACH_HEIGHT_MM,
                                vel=VEL_FAST, acc=ACC_FAST):
            return False
        time.sleep(0.3)

        # 4. Descend: 표면 위 3cm에서 release
        if not self.move_to_xyz(target[0], target[1],
                                target[2] + PLACE_RELEASE_HEIGHT_MM,
                                vel=VEL_SLOW, acc=ACC_SLOW):
            return False
        time.sleep(0.3)

        # 5. Gripper 열기 (release)
        self.robot.gripper_open()
        time.sleep(0.5)

        # 6. Retreat
        self.move_to_xyz(target[0], target[1],
                         target[2] + RETREAT_HEIGHT_MM,
                         vel=VEL_FAST, acc=ACC_FAST)
        print("[Place] SUCCESS")
        return True

    def pick_and_place(self, pick_instr, place_instr):
        """Pick & Place + Gemini 기반 retry (실패 시 재계획)"""
        for attempt in range(MAX_RETRIES):
            print(f"\n{'='*40} Attempt {attempt+1}/{MAX_RETRIES} "
                  f"{'='*40}")

            # Pick
            if not self.pick(pick_instr):
                # Gemini에게 실패 이미지를 보여주고 재계획 요청
                color, _ = self.camera.read()
                if color is not None:
                    replan = replan_after_failure(
                        color,
                        f"Pick failed (attempt {attempt+1}). "
                        f"The gripper missed the object for: "
                        f"{pick_instr}")
                    if replan:
                        print(f"[Gemini] Replan: {replan.get('adjustment', '')}")
                self.robot.gripper_open()
                self.robot.go_home()
                time.sleep(1.0)
                continue

            # Place
            if not self.place(place_instr):
                self.robot.gripper_open()
                self.robot.go_home()
                time.sleep(1.0)
                continue

            # Progress check: Gemini가 성공 여부 판단
            color, _ = self.camera.read()
            if color is not None:
                prog = check_progress(color, place_instr)
                if prog.get("complete"):
                    print(f"\nDone: {prog.get('reason', '')}")
                    return True
                print(f"[Gemini] Not complete: {prog.get('reason', '')}")

            self.robot.gripper_open()
            self.robot.go_home()
            time.sleep(1.0)

        print(f"\nFailed after {MAX_RETRIES} attempts")
        return False

    def run_single(self, instruction):
        """단일 pick & place"""
        print(f"\n[Task] {instruction}\n")
        return self.pick_and_place(instruction, instruction)

    def run_multi(self, instruction):
        """복합 명령 → Gemini가 분해 → 순차 실행"""
        print(f"\n[Task] {instruction}\n")
        color, _ = self.camera.read()
        steps = decompose_task(color, instruction) if color is not None else None
        if not steps:
            print("[Gemini] Decomposition failed")
            return False

        print(f"[Plan] {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"  {i+1}. {s['action']} -> {s['target']}")

        for i, step in enumerate(steps):
            print(f"\n--- Step {i+1}/{len(steps)}: "
                  f"{step['action']} {step['target']} ---")

            if step["action"] == "pick":
                ok = self.pick(step["target"])
            elif step["action"] == "place":
                ok = self.place(step["target"])
            else:
                print(f"[Unknown action: {step['action']}]")
                ok = False

            if not ok:
                print(f"[Step {i+1}] Failed, recovering...")
                self.robot.gripper_open()
                self.robot.go_home()
                time.sleep(1.0)

        print(f"\n{len(steps)} steps attempted")
        return True

    def run_icl(self, instruction, examples_dir="./track2/icl_examples"):
        """
        In-Context Learning 모드: few-shot waypoint 예시로 action 예측.

        teach pendant로 5~10개 waypoint에서 사진+TCP 기록 → examples/ 에 저장.
        Gemini가 context 안의 예시를 참고해서 다음 action을 step-by-step 예측.

        examples_dir 구조:
          icl_examples/
          ├── example_001.json  # {"tcp": [x,y,z,rx,ry,rz], "action": "...", "description": "..."}
          ├── example_001.jpg   # 해당 시점 카메라 이미지
          ├── example_002.json
          ├── example_002.jpg
          └── ...
        """
        import glob

        print(f"\n[ICL Mode] {instruction}")
        print(f"[ICL] Loading examples from {examples_dir}...")

        # Load examples
        examples = []
        json_files = sorted(glob.glob(f"{examples_dir}/example_*.json"))
        for jf in json_files:
            with open(jf) as f:
                ex = json.load(f)
            img_path = jf.replace(".json", ".jpg")
            ex["image_path"] = img_path
            examples.append(ex)
        print(f"[ICL] {len(examples)} examples loaded")

        if not examples:
            print("[ICL] No examples found! Record waypoints first.")
            print(f"[ICL] Expected: {examples_dir}/example_001.json + .jpg")
            return False

        # Action executor: parse function calls and execute
        def execute_action(action_str):
            """Gemini가 생성한 action string을 파싱해서 실행"""
            action_str = action_str.strip()
            try:
                if action_str.startswith("move_to("):
                    # parse: move_to(x, y, z, rx, ry, rz)
                    args_str = action_str[len("move_to("):-1]
                    args = [float(a.strip()) for a in args_str.split(",")]
                    x, y, z = args[0], args[1], args[2]
                    rx = args[3] if len(args) > 3 else 0.0
                    ry = args[4] if len(args) > 4 else 180.0
                    rz = args[5] if len(args) > 5 else 0.0
                    return self.robot.move_line(x, y, z, rx, ry, rz,
                                                vel=VEL_SLOW, acc=ACC_SLOW)
                elif action_str == "gripper_open()":
                    self.robot.gripper_open()
                    return True
                elif action_str == "gripper_close()":
                    self.robot.gripper_close()
                    return self.robot.is_object_grasped()
                elif action_str == "go_home()":
                    return self.robot.go_home()
                else:
                    print(f"[ICL] Unknown action: {action_str}")
                    return False
            except Exception as e:
                print(f"[ICL] Action exec error: {e}")
                return False

        # Run episode
        max_steps = 20
        for step in range(max_steps):
            color, _ = self.camera.read()
            if color is None:
                continue

            # 현재 TCP는 MoveL 모드에서 직접 읽기 어려우므로
            # joint → forward kinematics 대신, 마지막 MoveL 좌표를 추적하거나
            # MOCK에서는 [0,0,0,0,180,0]을 사용
            tcp = [0.0, 0.0, 0.0, 0.0, 180.0, 0.0]  # TODO: 실제 TCP 읽기

            result = predict_next_action_icl(
                color, tcp, instruction, examples)
            if result is None:
                print(f"[ICL] Step {step+1}: prediction failed, skipping")
                continue

            action = result.get("action", "")
            reasoning = result.get("reasoning", "")
            print(f"\n[ICL] Step {step+1}: {action}")
            print(f"       Reason: {reasoning}")

            if "go_home" in action:
                execute_action(action)
                print("[ICL] Episode complete")
                break

            execute_action(action)
            time.sleep(0.5)

        return True

    def cleanup(self):
        self.camera.release()
        self.robot.go_home()


# ===== ICL Example Recorder =====
def record_icl_examples(camera_index=0, output_dir="./track2/icl_examples"):
    """
    ICL 예시 수집 도구.
    teach pendant로 로봇을 이동시키며 각 waypoint에서 사진 + TCP를 기록.
    5~10개 waypoint만 기록하면 됨 (teleop 데모가 아님, 5~10분 소요).

    절차:
      1. teach pendant로 로봇을 원하는 포즈로 이동
      2. Enter → 카메라 이미지 저장 + TCP 좌표 입력
      3. 다음 행동(action)과 이유(description)를 입력
      4. 반복 (5~10회)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    idx = 1

    print(f"\n{'='*50}")
    print(f"  ICL Example Recorder")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")
    print(f"\nteach pendant로 로봇을 이동 → Enter로 기록")
    print(f"'q'로 종료\n")

    while True:
        cmd = input(f"[{idx}] 로봇 포즈 잡고 Enter (q=종료): ").strip()
        if cmd.lower() == "q":
            break

        # 카메라 캡처
        ret, frame = cap.read()
        if not ret:
            print("  Camera failed!")
            continue

        # TCP 좌표 입력
        tcp_str = input("  TCP [x,y,z,rx,ry,rz] (mm,deg): ").strip()
        try:
            tcp = [float(v) for v in tcp_str.split(",")]
            assert len(tcp) == 6
        except (ValueError, AssertionError):
            print("  Invalid TCP! Example: 400,0,300,0,180,0")
            continue

        # Action 입력
        action = input("  Action (e.g. move_to(400,0,250,0,180,0)): ").strip()
        description = input("  Why? (e.g. descend to reach bottle): ").strip()

        # 저장
        img_path = f"{output_dir}/example_{idx:03d}.jpg"
        json_path = f"{output_dir}/example_{idx:03d}.json"
        cv2.imwrite(img_path, frame)
        with open(json_path, "w") as f:
            json.dump({
                "tcp": tcp,
                "action": action,
                "description": description,
            }, f, indent=2)

        print(f"  Saved: {img_path} + {json_path}")
        idx += 1

    cap.release()
    print(f"\n{idx-1} examples saved to {output_dir}")


# ===== Test Modes =====
def test_pointing(camera_index=0):
    """Gemini pointing만 테스트 (로봇 불필요)"""
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Camera failed")
        return
    point = get_pick_point(frame, "the bottle closest to the robot")
    print(f"Pick point: {point}")
    if point:
        vis = frame.copy()
        cv2.circle(vis, point, 12, (0, 255, 0), 3)
        cv2.putText(vis, f"({point[0]},{point[1]})", (point[0]+15, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite("test_pointing.jpg", vis)
        print("Saved test_pointing.jpg")


def test_calib():
    """캘리브레이션 검증"""
    try:
        T = np.load(CALIBRATION_PATH)
        print(f"T_cam2base loaded:\n{T}")
        print(f"\nRotation det: {np.linalg.det(T[:3,:3]):.6f} (should be ~1.0)")
    except FileNotFoundError:
        print(f"{CALIBRATION_PATH} not found — run calibration.py first!")


def test_movel():
    """MoveL 단독 테스트 (안전한 위치로)"""
    robot = DoosanInterface()
    print("\n[Test] MoveL to safe position...")
    robot.move_line(400.0, 0.0, 300.0, 0.0, 180.0, 0.0,
                    vel=VEL_SLOW, acc=ACC_SLOW)
    time.sleep(1.0)
    print("[Test] Gripper open/close cycle...")
    robot.gripper_open()
    robot.gripper_close()
    print(f"[Test] Stroke: {robot.gripper_stroke}")
    print("[Test] Home...")
    robot.go_home()
    print("[Test] Done")


# ===== Main =====
def main():
    p = argparse.ArgumentParser(
        description="Track 2: Gemini-ER + MoveL + RGBD (Zero-shot)")
    p.add_argument("--instruction",
                   default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--mode", choices=["single", "multi", "icl"],
                   default="single",
                   help="single: pointing 기반, multi: task 분해, "
                        "icl: few-shot in-context learning")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--test-pointing", action="store_true",
                   help="Gemini pointing만 테스트")
    p.add_argument("--test-calib", action="store_true",
                   help="캘리브레이션 검증")
    p.add_argument("--test-movel", action="store_true",
                   help="MoveL + Gripper 테스트")
    p.add_argument("--record-icl", action="store_true",
                   help="ICL 예시 수집 (teach pendant + 카메라)")
    p.add_argument("--icl-examples-dir", default="./track2/icl_examples",
                   help="ICL 예시 디렉토리")
    args = p.parse_args()

    if args.test_pointing:
        test_pointing(args.camera_index)
        return
    if args.test_calib:
        test_calib()
        return
    if args.test_movel:
        test_movel()
        return
    if args.record_icl:
        record_icl_examples(args.camera_index, args.icl_examples_dir)
        return

    pipeline = Track2Pipeline(args)
    try:
        if args.mode == "multi":
            pipeline.run_multi(args.instruction)
        elif args.mode == "icl":
            pipeline.run_icl(args.instruction, args.icl_examples_dir)
        else:
            pipeline.run_single(args.instruction)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
