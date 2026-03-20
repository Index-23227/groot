"""
두산 E0509 데모 녹화기 — Direct Teaching (Hand-Guiding) + 카메라 동기 녹화

2인 체제:
  P1 (조작자): hand-guiding 버튼 누르고 로봇 팔 잡고 동작 수행
  P2 (운영자): 이 스크립트 실행, Enter/Ctrl+C로 녹화 제어, 물체 리셋

사용법:
  python utils/doosan_recorder.py --task "Pick up the blue bottle" --num-episodes 50
  python utils/doosan_recorder.py --verify --data-dir ./data/raw
"""

import os, sys, time, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *


class CameraCapture:
    def __init__(self):
        if CAMERA_TYPE == "opencv":
            import cv2
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            print(f"[Camera] OpenCV device {CAMERA_INDEX}")
        elif CAMERA_TYPE == "realsense":
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.rgb8, CAMERA_FPS)
            self.pipeline.start(cfg)
            print(f"[Camera] RealSense")
        else:
            print(f"[Camera] Using ROS2 topic: {CAMERA_TOPIC}")

    def read(self):
        if CAMERA_TYPE == "opencv":
            import cv2
            ret, frame = self.cap.read()
            if not ret:
                return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif CAMERA_TYPE == "realsense":
            frames = self.pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def release(self):
        if CAMERA_TYPE == "opencv" and hasattr(self, 'cap'):
            self.cap.release()
        elif CAMERA_TYPE == "realsense" and hasattr(self, 'pipeline'):
            self.pipeline.stop()


class RobotStateReader:
    def __init__(self):
        self.latest_joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.latest_gripper = 0.0
        self.connected = False
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            try:
                rclpy.init()
            except RuntimeError:
                pass
            self.node = rclpy.create_node("doosan_recorder")
            topic = f"/dsr01{ROBOT_MODEL}/joint_states"
            self.node.create_subscription(JointState, topic, self._cb, 10)
            self.connected = True
            print(f"[Robot] ROS2: {topic}")
        except ImportError:
            print("[Robot] ROS2 unavailable — MOCK mode")

    def _cb(self, msg):
        self.latest_joints = np.array(msg.position[:NUM_JOINTS], dtype=np.float32)

    def read(self):
        if self.connected:
            import rclpy
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.latest_joints.copy(), self.latest_gripper

    def shutdown(self):
        if self.connected:
            import rclpy
            self.node.destroy_node()
            rclpy.shutdown()


class DemoRecorder:
    def __init__(self, save_dir, task, fps=10.0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.camera = CameraCapture()
        self.robot = RobotStateReader()
        self.episode_count = 0

    def record_episode(self):
        ep_id = self.episode_count
        print(f"\n{'=' * 50}")
        print(f"  Episode {ep_id}")
        print(f"  Task: {self.task}")
        print(f"  P1: hand-guiding 버튼 누르고 동작 수행")
        print(f"  P2: [Enter] 시작 → [Ctrl+C] 종료")
        print(f"{'=' * 50}")
        input("Enter to START...")

        states, images, timestamps = [], [], []
        print("🔴 Recording...")

        try:
            while True:
                t0 = time.time()
                joints, gripper = self.robot.read()
                state = np.concatenate([joints, [gripper]]).astype(np.float32)
                states.append(state)
                images.append(self.camera.read())
                timestamps.append(time.time())
                elapsed = time.time() - t0
                if self.dt - elapsed > 0:
                    time.sleep(self.dt - elapsed)
        except KeyboardInterrupt:
            pass

        n = len(states)
        print(f"⏹️  {n} frames")

        if n < 10:
            print("⚠️ Too few frames, skipping.")
            return None

        states_arr = np.array(states)
        actions = np.vstack([np.diff(states_arr, axis=0), np.zeros((1, ACTION_DIM))])

        ep_dir = self.save_dir / f"episode_{ep_id:04d}"
        ep_dir.mkdir(exist_ok=True)
        np.savez_compressed(str(ep_dir / "data.npz"), states=states_arr, actions=actions)
        np.savez_compressed(str(ep_dir / "images.npz"), images=np.array(images))
        meta = {"episode_id": ep_id, "task": self.task, "num_frames": n,
                "fps": self.fps, "robot": ROBOT_MODEL,
                "recorded_at": datetime.now().isoformat()}
        with open(str(ep_dir / "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"💾 {ep_dir}")
        self.episode_count += 1
        return meta

    def finish(self):
        self.camera.release()
        self.robot.shutdown()
        print(f"\n✅ {self.episode_count} episodes in {self.save_dir}")


def verify_data(data_dir):
    episodes = sorted(Path(data_dir).glob("episode_*"))
    print(f"Found {len(episodes)} episodes")
    for ep_dir in episodes:
        try:
            data = np.load(str(ep_dir / "data.npz"))
            with open(str(ep_dir / "metadata.json")) as f:
                meta = json.load(f)
            n = meta["num_frames"]
            s, a = data["states"], data["actions"]
            ok = s.shape == (n, ACTION_DIM) and a.shape == (n, ACTION_DIM)
            status = "✅" if ok else "❌"
            print(f"  {status} {ep_dir.name}: {n}f, state[{s.min():.2f},{s.max():.2f}]")
        except Exception as e:
            print(f"  ❌ {ep_dir.name}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default=RAW_DATA_DIR)
    p.add_argument("--task", default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--fps", type=float, default=float(CONTROL_HZ))
    p.add_argument("--verify", action="store_true")
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    if args.verify:
        verify_data(args.data_dir or args.save_dir)
    else:
        rec = DemoRecorder(args.save_dir, args.task, args.fps)
        for i in range(args.num_episodes):
            rec.record_episode()
            if i < args.num_episodes - 1:
                c = input(f"\n다음? ({i+1}/{args.num_episodes}) [Enter/q]: ")
                if c.strip().lower() == "q":
                    break
        rec.finish()
