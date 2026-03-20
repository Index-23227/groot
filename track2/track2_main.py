"""
Track 2: Gemini Robotics-ER 1.5 + cuRobo + RGBD
Zero-shot Pick & Place Pipeline

Usage:
  # 단일 태스크
  python track2_main.py --instruction "파란 약병을 조제함에 넣어"

  # 복합 태스크 (Gemini-ER가 자동 분해)
  python track2_main.py --instruction "약병들을 색깔별로 분류해" --mode multi

  # 모듈 테스트
  python track2_main.py --test-pointing    # Gemini pointing만 테스트
  python track2_main.py --test-calib       # 캘리브레이션 검증
"""

import argparse
import time
import cv2
import numpy as np

from calibration import pixel_depth_to_robot, CAMERA_INTRINSICS
from gemini_er_client import (
    get_pick_point, get_place_point,
    decompose_task, check_progress,
)
from ik_solver import IKSolverWrapper

# ===== 현장에서 수정 =====
ROBOT_IP = "192.168.137.100"
URDF_PATH = "~/doosan_ws/src/e0509_gripper_description/urdf/e0509_with_gripper.urdf.xacro"
CALIBRATION_PATH = "T_cam2base.npy"

APPROACH_HEIGHT_MM = 100
RETREAT_HEIGHT_MM = 150
GRIPPER_CLOSE_WAIT = 1.5
MAX_RETRIES = 3
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ===== Robot =====
class DoosanInterface:
    def __init__(self):
        self.mock = True
        self.gripper_stroke = 0
        self.joint_positions = np.zeros(6)
        try:
            import rclpy
            from dsr_msgs2.srv import MoveJoint
            from std_srvs.srv import Trigger
            from std_msgs.msg import Int32
            from sensor_msgs.msg import JointState
            try:
                rclpy.init()
            except RuntimeError:
                pass
            self.node = rclpy.create_node("track2_controller")
            self.move_joint_cli = self.node.create_client(MoveJoint, "/dsr01/motion/move_joint")
            self.gripper_open_cli = self.node.create_client(Trigger, "/dsr01/gripper/open")
            self.gripper_close_cli = self.node.create_client(Trigger, "/dsr01/gripper/close")
            self.node.create_subscription(Int32, "/dsr01/gripper/stroke", self._stroke_cb, 10)
            self.node.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
            self.mock = False
            print("[Robot] ROS2 connected")
        except ImportError:
            print("[Robot] MOCK mode (ROS2 not available)")

    def _stroke_cb(self, msg):
        self.gripper_stroke = msg.data

    def _joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.joint_positions = np.array(msg.position[:6])

    def move_joint(self, joint_deg, vel=30.0, acc=30.0):
        if self.mock:
            print(f"[MOCK] MoveJoint → [{', '.join(f'{d:.1f}' for d in joint_deg)}]")
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

    def is_object_grasped(self, threshold=50):
        if self.mock:
            return self.gripper_stroke > threshold
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.1)
        return threshold < self.gripper_stroke < 650


# ===== Camera =====
class RGBDCamera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.depth = None
        # TODO: 실제 depth 카메라 초기화 (RealSense 등)
        print(f"[Camera] index {index}")

    def read(self):
        ret, color = self.cap.read()
        return color, self.depth

    def get_depth_at(self, x, y):
        if self.depth is None:
            return 600.0  # MOCK: 60cm 고정
        region = self.depth[max(0, y-2):y+3, max(0, x-2):x+3]
        valid = region[region > 0]
        return float(np.median(valid)) if len(valid) > 0 else None

    def release(self):
        self.cap.release()


# ===== Pipeline =====
class Track2Pipeline:
    def __init__(self, args):
        self.camera = RGBDCamera(args.camera_index)
        self.robot = DoosanInterface()
        try:
            self.ik = IKSolverWrapper(URDF_PATH)
        except Exception as e:
            print(f"[IK] Init failed: {e}")
            self.ik = None
        try:
            self.T_cam2base = np.load(CALIBRATION_PATH)
        except FileNotFoundError:
            print(f"[Calib] {CALIBRATION_PATH} not found — run calibration.py first!")
            self.T_cam2base = np.eye(4)

    def pixel_to_robot_3d(self, px, py, depth_mm):
        return pixel_depth_to_robot(px, py, depth_mm, CAMERA_INTRINSICS, self.T_cam2base)

    def move_to_3d(self, target_xyz_mm, approach=False):
        if self.ik is None:
            print("[IK] Solver not available!")
            return False
        if approach:
            above = target_xyz_mm.copy()
            above[2] += APPROACH_HEIGHT_MM
            joints = self.ik.solve(above)
            if joints:
                self.robot.move_joint(joints, vel=20.0, acc=20.0)
                time.sleep(0.5)
        joints = self.ik.solve(target_xyz_mm)
        if joints:
            self.robot.move_joint(joints, vel=10.0, acc=10.0)
            time.sleep(0.5)
            return True
        return False

    def pick(self, instruction):
        color, _ = self.camera.read()
        if color is None:
            return False
        print(f"\n[Gemini] Finding: {instruction}")
        point = get_pick_point(color, instruction)
        if point is None:
            return False
        px, py = point
        print(f"[Gemini] → pixel ({px}, {py})")
        depth = self.camera.get_depth_at(px, py)
        if depth is None:
            return False
        target = self.pixel_to_robot_3d(px, py, depth)
        print(f"[3D] → robot [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}] mm")
        self.robot.gripper_open()
        if not self.move_to_3d(target, approach=True):
            return False
        self.robot.gripper_close()
        if not self.robot.is_object_grasped():
            print("[Grasp] FAILED")
            return False
        lift = target.copy()
        lift[2] += RETREAT_HEIGHT_MM
        self.move_to_3d(lift)
        print("[Pick] ✅")
        return True

    def place(self, instruction):
        color, _ = self.camera.read()
        if color is None:
            return False
        point = get_place_point(color, instruction)
        if point is None:
            return False
        px, py = point
        depth = self.camera.get_depth_at(px, py)
        if depth is None:
            return False
        target = self.pixel_to_robot_3d(px, py, depth)
        target[2] += 30
        if not self.move_to_3d(target, approach=True):
            return False
        self.robot.gripper_open()
        time.sleep(0.5)
        retreat = target.copy()
        retreat[2] += RETREAT_HEIGHT_MM
        self.move_to_3d(retreat)
        print("[Place] ✅")
        return True

    def pick_and_place(self, pick_instr, place_instr):
        for attempt in range(MAX_RETRIES):
            print(f"\n{'='*40} Attempt {attempt+1}/{MAX_RETRIES} {'='*40}")
            if self.pick(pick_instr) and self.place(place_instr):
                color, _ = self.camera.read()
                if color is not None:
                    prog = check_progress(color, place_instr)
                    if prog.get("complete"):
                        print(f"\n✅ Done: {prog.get('reason','')}")
                        return True
            self.robot.gripper_open()
            self.robot.move_joint(HOME_POSITION)
            time.sleep(1.0)
        print(f"\n❌ Failed after {MAX_RETRIES} attempts")
        return False

    def run_single(self, instruction):
        print(f"\n📋 {instruction}\n")
        return self.pick_and_place(instruction, instruction)

    def run_multi(self, instruction):
        print(f"\n📋 {instruction}\n")
        color, _ = self.camera.read()
        steps = decompose_task(color, instruction) if color is not None else None
        if not steps:
            print("[Gemini] Decomposition failed")
            return False
        print(f"[Plan] {len(steps)} steps:")
        for i, s in enumerate(steps):
            print(f"  {i+1}. {s['action']} → {s['target']}")
        for i, step in enumerate(steps):
            print(f"\n--- Step {i+1}/{len(steps)}: {step['action']} {step['target']} ---")
            ok = self.pick(step["target"]) if step["action"] == "pick" else self.place(step["target"])
            if not ok:
                self.robot.gripper_open()
                self.robot.move_joint(HOME_POSITION)
        print(f"\n✅ {len(steps)} steps attempted")
        return True

    def cleanup(self):
        self.camera.release()
        self.robot.move_joint(HOME_POSITION)


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


# ===== Main =====
def main():
    p = argparse.ArgumentParser(description="Track 2: Gemini-ER + cuRobo + RGBD")
    p.add_argument("--instruction", default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--mode", choices=["single", "multi"], default="single")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--test-pointing", action="store_true", help="Gemini pointing만 테스트")
    p.add_argument("--test-calib", action="store_true", help="캘리브레이션 검증")
    args = p.parse_args()

    if args.test_pointing:
        test_pointing(args.camera_index)
        return
    if args.test_calib:
        test_calib()
        return

    pipeline = Track2Pipeline(args)
    try:
        if args.mode == "multi":
            pipeline.run_multi(args.instruction)
        else:
            pipeline.run_single(args.instruction)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
