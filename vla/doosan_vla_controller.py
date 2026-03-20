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
from vla.doosan_action_adapter import DoosanActionAdapter, DoosanSafetyConfig


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
    """실제 두산 E0509 + ROBOTIS RH-P12-RN-A 제어.

    /joint_states에서 arm 6 + gripper 상태를 읽고,
    MoveJoint 서비스(degree)와 gripper 서비스로 제어.
    """

    def __init__(self):
        self.connected = False
        self.latest_joints = np.zeros(NUM_JOINTS)
        self.latest_gripper = 0.0  # 0.0=열림, 1.0=닫힘
        self._grip_pub = None
        self._grip_open_client = None
        self._grip_close_client = None
        try:
            import rclpy
            from sensor_msgs.msg import JointState
            try: rclpy.init()
            except RuntimeError: pass
            self.node = rclpy.create_node("vla_controller")

            # Joint states 구독 (arm 6 + gripper 4)
            self.node.create_subscription(
                JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)

            # Gripper stroke 구독
            try:
                from std_msgs.msg import Int32
                self.node.create_subscription(
                    Int32, GRIPPER_STROKE_TOPIC, self._stroke_cb, 10)
                # Gripper stroke 연속 제어용 publisher
                self._grip_pub = self.node.create_publisher(
                    Int32, GRIPPER_POSITION_TOPIC, 10)
            except Exception:
                pass

            # Gripper open/close 서비스 클라이언트
            try:
                from std_srvs.srv import Trigger
                self._grip_open_client = self.node.create_client(
                    Trigger, GRIPPER_OPEN_SERVICE)
                self._grip_close_client = self.node.create_client(
                    Trigger, GRIPPER_CLOSE_SERVICE)
            except Exception:
                pass

            self.connected = True
            print(f"[Robot] ROS2 connected: {JOINT_STATE_TOPIC}")
        except ImportError:
            print("[Robot] MOCK mode")

    def _joint_cb(self, msg):
        self.latest_joints = np.array(msg.position[:NUM_JOINTS])
        if len(msg.position) > NUM_JOINTS:
            self.latest_gripper = float(msg.position[NUM_JOINTS])

    def _stroke_cb(self, msg):
        self.latest_gripper = stroke_to_grip(msg.data)

    def get_state(self):
        if self.connected:
            import rclpy
            rclpy.spin_once(self.node, timeout_sec=0.01)
        return self.latest_joints.copy(), self.latest_gripper

    def send(self, cmd, dt=0.1):
        """adapter.convert() 결과를 받아 로봇에 명령 전송.

        Args:
            cmd: adapter.convert()의 반환값 dict
        """
        deg = cmd["joint_targets_deg"].tolist()
        grip_close = cmd["gripper_close"]
        stroke = cmd["gripper_stroke"]

        if self.connected:
            # Gripper 제어: 연속(stroke) 또는 이진(open/close)
            if self._grip_pub is not None:
                from std_msgs.msg import Int32
                self._grip_pub.publish(Int32(data=stroke))
            elif grip_close and self._grip_close_client:
                from std_srvs.srv import Trigger
                self._grip_close_client.call_async(Trigger.Request())
            elif not grip_close and self._grip_open_client:
                from std_srvs.srv import Trigger
                self._grip_open_client.call_async(Trigger.Request())

        g = "CLOSE" if grip_close else "OPEN"
        print(f"[CMD] [{', '.join(f'{d:.1f}' for d in deg)}] {g}(stroke={stroke})")

    def send_legacy(self, joints_rad, gripper_open, dt=0.1):
        """하위 호환: 이전 인터페이스."""
        deg = np.rad2deg(joints_rad).tolist()
        g = "OPEN" if gripper_open else "CLOSE"
        print(f"[CMD] [{', '.join(f'{d:.1f}' for d in deg)}] {g}")


def main(args):
    from utils.doosan_recorder import CameraCapture
    from vla.failure_detector import FailureDetector

    vla = VLAClient(args.vla_url)
    robot = DoosanRobot()
    adapter = DoosanActionAdapter()
    camera = CameraCapture()
    blender = TemporalBlender(
        execute_horizon=args.execute_horizon,
        overlap=args.overlap,
        decay=args.decay,
    )
    detector = FailureDetector()
    dt = 1.0 / args.hz

    # --- Level 2: STT instruction ---
    instruction = args.instruction
    if args.stt:
        from utils.stt_instruction import STTInstruction
        stt = STTInstruction(use_llm=args.stt_llm)
        heard = stt.listen(duration=args.stt_duration)
        if heard:
            instruction = heard
        else:
            print("[STT] 인식 실패, 기본 instruction 사용")

    print(f"\n=== Deploy: {instruction} @ {args.hz}Hz ===")
    print(f"    chunk: execute_horizon={args.execute_horizon}, overlap={args.overlap}, decay={args.decay}\n")

    step = 0
    while step < args.max_steps:
        # VLA inference — action chunk 획득
        joints, grip = robot.get_state()
        adapter.set_current_state(joints, grip)
        img = camera.read()
        chunk = vla.predict(img, np.concatenate([joints, [grip]]), instruction)
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
            robot.send(cmd, dt)

            # --- Level 4: 실패 감지 ---
            status = detector.update(cmd["joint_targets_rad"], cmd["clamp_ratio"])
            if status["should_fallback"]:
                print(f"\n⚠️  [step {step}] 반복 실패 — classical fallback 전환")
                camera.release()
                from vla.plan_c_classical import main as classical_main
                classical_main(argparse.Namespace(
                    instruction=instruction, test_vision=False))
                return
            if status["should_retry"]:
                reason = "stall" if status["stalled"] else "over-clamp"
                print(f"\n🔄 [step {step}] {reason} 감지 — 재시도 #{detector.retry_count}")
                blender.reset()
                break  # 현재 chunk 중단, 새 inference부터 재시작

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
    # Level 2: STT
    p.add_argument("--stt", action="store_true", help="음성으로 instruction 입력")
    p.add_argument("--stt-llm", action="store_true", help="LLM으로 instruction 정제")
    p.add_argument("--stt-duration", type=float, default=5)
    main(p.parse_args())
