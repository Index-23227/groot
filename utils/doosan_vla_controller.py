"""
배포: VLA Inference Server ↔ Action Adapter ↔ Robot 제어 루프

사용법:
  python utils/doosan_vla_controller.py --vla-url http://localhost:5555 \
    --instruction "Pick up the blue bottle and place it on the tray"
"""

import os, sys, time, argparse
import numpy as np
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *
from utils.doosan_action_adapter import DoosanActionAdapter, DoosanSafetyConfig


class VLAClient:
    def __init__(self, url):
        self.url = url

    def predict(self, image, state, instruction):
        import base64
        from io import BytesIO
        from PIL import Image
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=85)
        try:
            r = requests.post(f"{self.url}/predict",
                json={"image": base64.b64encode(buf.getvalue()).decode(),
                      "state": state.tolist(), "instruction": instruction}, timeout=3)
            r.raise_for_status()
            return np.array(r.json()["actions"])
        except Exception as e:
            print(f"[VLA] {e}")
            return None


class DoosanRobot:
    def __init__(self):
        self.connected = False
        self.latest_joints = np.zeros(NUM_JOINTS)
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            try: rclpy.init()
            except RuntimeError: pass
            self.node = rclpy.create_node("vla_controller")
            self.node.create_subscription(JointState,
                f"/dsr01{ROBOT_MODEL}/joint_states", self._cb, 10)
            self.connected = True
        except ImportError:
            print("[Robot] MOCK mode")

    def _cb(self, msg):
        self.latest_joints = np.array(msg.position[:NUM_JOINTS])

    def get_state(self):
        if self.connected:
            import rclpy
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.latest_joints.copy(), 0.0

    def send(self, joints, gripper_open, dt=0.1):
        deg = np.rad2deg(joints).tolist()
        g = "OPEN" if gripper_open else "CLOSE"
        print(f"[CMD] [{', '.join(f'{d:.1f}' for d in deg)}] {g}")


def main(args):
    from utils.doosan_recorder import CameraCapture
    vla = VLAClient(args.vla_url)
    robot = DoosanRobot()
    adapter = DoosanActionAdapter()
    camera = CameraCapture()
    dt = 1.0 / args.hz

    print(f"\n=== Deploy: {args.instruction} @ {args.hz}Hz ===\n")
    for step in range(args.max_steps):
        t0 = time.time()
        joints, grip = robot.get_state()
        adapter.set_current_state(joints, grip)
        img = camera.read()
        act = vla.predict(img, np.concatenate([joints, [grip]]), args.instruction)
        if act is None: continue
        if act.ndim == 2: act = act[0]
        cmd = adapter.convert(act, dt)
        robot.send(cmd["joint_targets"], cmd["gripper_open"], dt)
        elapsed = time.time() - t0
        if dt - elapsed > 0: time.sleep(dt - elapsed)
        if step % 20 == 0:
            print(f"  [step {step}] clamp={cmd['clamp_ratio']:.0%} dt={elapsed*1000:.0f}ms")
    camera.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default="http://localhost:5555")
    p.add_argument("--instruction", default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--hz", type=float, default=float(CONTROL_HZ))
    p.add_argument("--max-steps", type=int, default=200)
    main(p.parse_args())
