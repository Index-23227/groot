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


class TemporalBlender:
    """Action chunk temporal blending — 연속 chunk 간 부드러운 전환.

    VLA 모델이 매 스텝 action chunk (T, action_dim)을 출력할 때,
    이전 잔여 action과 새 chunk를 overlap 구간에서 가중 평균하여
    급격한 action 변화를 완화한다.
    """

    def __init__(self, execute_horizon: int = 4, overlap: int = 4, decay: float = 0.7):
        self.execute_horizon = execute_horizon
        self.overlap = overlap
        self.decay = decay
        self._buffer = None  # 이전 chunk의 잔여 action

    def blend(self, chunk: np.ndarray) -> np.ndarray:
        """chunk (T, action_dim) → 실행할 actions (execute_horizon, action_dim)"""
        if self._buffer is None:
            # 첫 chunk: blending 없이 그대로 반환
            self._buffer = chunk[self.execute_horizon:]
            return chunk[:self.execute_horizon].copy()

        # overlap 구간에서 이전 잔여와 새 chunk를 blending
        new = chunk.copy()
        blend_len = min(self.overlap, len(self._buffer), len(new))
        for i in range(blend_len):
            # decay^(i+1): 앞쪽일수록 이전 버퍼 비중 높음, 뒤로 갈수록 새 chunk 비중 높음
            w_old = self.decay ** (i + 1)
            w_new = 1.0 - w_old
            new[i] = w_old * self._buffer[i] + w_new * new[i]

        result = new[:self.execute_horizon]
        self._buffer = new[self.execute_horizon:]
        return result

    def reset(self):
        self._buffer = None


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
    blender = TemporalBlender(
        execute_horizon=args.execute_horizon,
        overlap=args.overlap,
        decay=args.decay,
    )
    dt = 1.0 / args.hz

    print(f"\n=== Deploy: {args.instruction} @ {args.hz}Hz ===")
    print(f"    chunk: execute_horizon={args.execute_horizon}, overlap={args.overlap}, decay={args.decay}\n")

    step = 0
    while step < args.max_steps:
        # VLA inference — action chunk 획득
        joints, grip = robot.get_state()
        adapter.set_current_state(joints, grip)
        img = camera.read()
        chunk = vla.predict(img, np.concatenate([joints, [grip]]), args.instruction)
        if chunk is None:
            continue
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        # Temporal blending → execute_horizon 개의 action
        actions = blender.blend(chunk)

        # execute_horizon 스텝 동안 action 실행
        for act in actions:
            if step >= args.max_steps:
                break
            t0 = time.time()
            joints, grip = robot.get_state()
            adapter.set_current_state(joints, grip)
            cmd = adapter.convert(act, dt)
            robot.send(cmd["joint_targets"], cmd["gripper_open"], dt)
            elapsed = time.time() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
            if step % 20 == 0:
                print(f"  [step {step}] clamp={cmd['clamp_ratio']:.0%} dt={elapsed*1000:.0f}ms")
            step += 1
    camera.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default="http://localhost:5555")
    p.add_argument("--instruction", default="Pick up the blue bottle and place it on the tray")
    p.add_argument("--hz", type=float, default=float(CONTROL_HZ))
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--execute-horizon", type=int, default=4)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--decay", type=float, default=0.7)
    main(p.parse_args())
