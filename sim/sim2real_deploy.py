"""
Sim-to-Real 배포 — 시뮬에서 학습한 모델을 실제 E0509 로봇에 배포

학습 파이프라인:
  1. Isaac Sim 데모 수집 (sim/isaac_data_collector.py)
  2. LeRobot v2 변환 (utils/convert_to_lerobot.py)
  3. GR00T/SmolVLA 학습 (scripts/04_train_groot.sh)
  4. → 이 스크립트로 실제 로봇에 배포

Sim-to-Real 보정:
  - Action scale 보정 (sim과 real의 동작 크기 차이)
  - Observation normalization 맞추기
  - 실제 카메라 이미지에 sim 스타일 augmentation 역적용

사용법:
  # Isaac Sim에서 검증 (sim → sim)
  python sim/sim2real_deploy.py --mode sim-eval \
    --checkpoint ./checkpoints/groot/checkpoint-10000

  # 실제 로봇 배포 (sim → real)
  python sim/sim2real_deploy.py --mode real \
    --checkpoint ./checkpoints/groot/checkpoint-10000 \
    --instruction "Pick up the blue medicine bottle and place it in dispensing slot 1"

  # Action scale 캘리브레이션
  python sim/sim2real_deploy.py --mode calibrate
"""

import os, sys, time, argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *
from configs.pharmacy_scenario import SCENARIOS, get_scenario
from utils.doosan_action_adapter import DoosanActionAdapter
from utils.doosan_vla_controller import TemporalBlender, VLAClient, DoosanRobot
from utils.failure_detector import FailureDetector


# =============================================================
# Sim-to-Real 보정
# =============================================================

class Sim2RealConfig:
    """Sim ↔ Real 차이 보정 파라미터

    시뮬레이션과 실제 로봇 사이의 gap을 보정.
    캘리브레이션 후 이 값들을 업데이트.
    """

    def __init__(self):
        # Action scale: sim에서 학습한 action이 real에서 얼마나 커야 하는지
        # scale > 1.0 → sim action이 real에서 더 크게 적용
        # scale < 1.0 → sim action이 real에서 더 작게 적용
        self.action_scale = np.ones(NUM_JOINTS)

        # Joint offset: sim과 real의 zero position 차이 (rad)
        self.joint_offset = np.zeros(NUM_JOINTS)

        # Gripper: sim과 real의 gripper 매핑
        self.gripper_threshold = GRIPPER_THRESHOLD

        # Image: sim → real 보정 (밝기, 대비)
        self.image_brightness_offset = 0     # -50 ~ +50
        self.image_contrast_scale = 1.0      # 0.5 ~ 1.5

        # Safety: real에서 더 보수적인 safety margin
        self.real_max_delta = MAX_DELTA_PER_STEP * 0.7  # sim 대비 70%로 제한
        self.real_max_velocity = MAX_JOINT_VELOCITY * 0.5  # sim 대비 50%

    def save(self, path):
        """보정값 저장"""
        data = {
            "action_scale": self.action_scale.tolist(),
            "joint_offset": self.joint_offset.tolist(),
            "gripper_threshold": self.gripper_threshold,
            "image_brightness_offset": self.image_brightness_offset,
            "image_contrast_scale": self.image_contrast_scale,
            "real_max_delta": self.real_max_delta,
            "real_max_velocity": self.real_max_velocity,
        }
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Sim2Real] Config saved: {path}")

    def load(self, path):
        """보정값 로드"""
        import json
        with open(path) as f:
            data = json.load(f)
        self.action_scale = np.array(data["action_scale"])
        self.joint_offset = np.array(data["joint_offset"])
        self.gripper_threshold = data["gripper_threshold"]
        self.image_brightness_offset = data["image_brightness_offset"]
        self.image_contrast_scale = data["image_contrast_scale"]
        self.real_max_delta = data["real_max_delta"]
        self.real_max_velocity = data["real_max_velocity"]
        print(f"[Sim2Real] Config loaded: {path}")

    def transform_action(self, action):
        """Sim에서 학습한 action → Real 로봇용으로 변환"""
        transformed = action.copy()
        # Joint action에 scale 적용
        transformed[:NUM_JOINTS] = action[:NUM_JOINTS] * self.action_scale
        return transformed

    def transform_state(self, state):
        """Real 로봇 state → Sim 학습 모델 입력용으로 변환"""
        transformed = state.copy()
        # Joint offset 보정 (real → sim 좌표계)
        transformed[:NUM_JOINTS] = state[:NUM_JOINTS] - self.joint_offset
        return transformed

    def transform_image(self, image):
        """Real 카메라 이미지 → Sim 학습 모델 입력용으로 변환"""
        img = image.astype(np.float32)
        # 밝기 보정
        img += self.image_brightness_offset
        # 대비 보정
        mean = img.mean()
        img = (img - mean) * self.image_contrast_scale + mean
        return np.clip(img, 0, 255).astype(np.uint8)


# =============================================================
# Sim-to-Real 배포 컨트롤러
# =============================================================

class Sim2RealController:
    """시뮬에서 학습한 모델 → 실제 로봇 배포"""

    def __init__(self, args):
        self.args = args
        self.s2r_config = Sim2RealConfig()

        # 보정 파일이 있으면 로드
        config_path = os.path.join(os.path.dirname(args.checkpoint), "sim2real_config.json")
        if os.path.exists(config_path):
            self.s2r_config.load(config_path)

        # VLA inference client
        self.vla = VLAClient(args.vla_url)

        # Action adapter (sim2real 보정된 safety)
        from utils.doosan_action_adapter import DoosanSafetyConfig
        safety = DoosanSafetyConfig(
            max_delta_per_step=self.s2r_config.real_max_delta,
            max_joint_velocity=self.s2r_config.real_max_velocity,
        )
        self.adapter = DoosanActionAdapter(safety_config=safety)

        # Temporal blender
        self.blender = TemporalBlender(
            execute_horizon=args.execute_horizon,
            overlap=args.overlap,
            decay=args.decay,
        )

        # Failure detector
        self.detector = FailureDetector()

        # Control params
        self.dt = 1.0 / args.hz

    def run_real(self, instruction):
        """실제 로봇에서 실행"""
        from utils.doosan_recorder import CameraCapture

        robot = DoosanRobot()
        camera = CameraCapture()

        print(f"\n{'='*60}")
        print(f"  Sim-to-Real Deploy")
        print(f"  Model: {self.args.checkpoint}")
        print(f"  Instruction: {instruction}")
        print(f"  Action scale: [{', '.join(f'{s:.2f}' for s in self.s2r_config.action_scale)}]")
        print(f"  Safety: delta={np.rad2deg(self.s2r_config.real_max_delta):.1f}°, "
              f"vel={np.rad2deg(self.s2r_config.real_max_velocity):.1f}°/s")
        print(f"{'='*60}\n")

        input("⚠️  로봇 주변 안전 확인 후 [Enter]를 누르세요...")

        step = 0
        while step < self.args.max_steps:
            # 로봇 상태 (real → sim 좌표계 변환)
            joints, grip = robot.get_state()
            state_for_model = self.s2r_config.transform_state(
                np.concatenate([joints, [grip]])
            )
            self.adapter.set_current_state(joints, grip)

            # 카메라 이미지 (real → sim 스타일 변환)
            img = camera.read()
            img_for_model = self.s2r_config.transform_image(img)

            # VLA inference
            chunk = self.vla.predict(img_for_model, state_for_model, instruction)
            if chunk is None:
                continue
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            # Sim action → Real action 변환
            for i in range(len(chunk)):
                chunk[i] = self.s2r_config.transform_action(chunk[i])

            # Temporal blending
            actions = self.blender.blend(chunk)

            # 실행
            chunk_interrupted = False
            for act in actions:
                if step >= self.args.max_steps:
                    break

                t0 = time.time()
                joints, grip = robot.get_state()
                self.adapter.set_current_state(joints, grip)
                cmd = self.adapter.convert(act, self.dt)
                robot.send(cmd["joint_targets"], cmd["gripper_open"], self.dt)

                # 실패 감지
                status = self.detector.update(cmd["joint_targets"], cmd["clamp_ratio"])

                if status["should_fallback"]:
                    print(f"\n⚠️  [step {step}] Sim2Real 실패 — classical fallback")
                    camera.release()
                    from utils.plan_c_classical import main as classical_main
                    classical_main(argparse.Namespace(
                        instruction=instruction, test_vision=False))
                    return

                if status["should_retry"]:
                    reason = "stall" if status["stalled"] else "over-clamp"
                    print(f"\n🔄 [step {step}] {reason} 감지 — 재시도 #{self.detector.retry_count}")
                    self.blender.reset()
                    chunk_interrupted = True
                    break

                elapsed = time.time() - t0
                if self.dt - elapsed > 0:
                    time.sleep(self.dt - elapsed)

                if step % 20 == 0:
                    joints_deg = np.rad2deg(cmd["joint_targets"])
                    grip_str = "OPEN" if cmd["gripper_open"] else "CLOSE"
                    print(f"  [step {step:3d}] [{', '.join(f'{d:.1f}' for d in joints_deg)}] "
                          f"{grip_str}  clamp={cmd['clamp_ratio']:.0%}")
                step += 1

        camera.release()
        print(f"\n✅ 완료: {step} steps")

    def run_sim_eval(self, scenario):
        """Isaac Sim에서 학습된 모델 평가 (real 배포 전 검증)"""
        from sim.pharmacy_isaac_env import PharmacyIsaacEnv

        env = PharmacyIsaacEnv(headless=self.args.headless)

        print(f"\n{'='*60}")
        print(f"  Sim Evaluation")
        print(f"  Model: {self.args.checkpoint}")
        print(f"  Scenario: {scenario['id']}")
        print(f"{'='*60}\n")

        try:
            obs = env.reset()
            instruction = scenario["instruction"]
            step = 0

            while step < self.args.max_steps:
                state = obs["state"]
                image = obs["image"]
                self.adapter.set_current_state(state[:NUM_JOINTS], state[NUM_JOINTS])

                chunk = self.vla.predict(image, state, instruction)
                if chunk is None:
                    continue
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)

                actions = self.blender.blend(chunk)

                for act in actions:
                    if step >= self.args.max_steps:
                        break
                    obs, reward, done, info = env.step(act)
                    step += 1

                    if step % 20 == 0:
                        joints_deg = np.rad2deg(obs["state"][:NUM_JOINTS])
                        print(f"  [step {step:3d}] [{', '.join(f'{d:.1f}' for d in joints_deg)}]")

            # 성공 판정
            success = env.is_object_in_slot(scenario["pick"], scenario["place"])
            print(f"\n  판정: {'✅ SUCCESS' if success else '❌ FAIL'}")
            return success

        finally:
            env.close()


# =============================================================
# 캘리브레이션
# =============================================================

def calibrate(args):
    """Sim ↔ Real 보정값 측정

    1. 실제 로봇의 home position 측정
    2. 몇 가지 기준 동작을 sim/real 양쪽에서 실행
    3. 차이를 action_scale, joint_offset으로 기록
    """
    from utils.doosan_recorder import CameraCapture

    s2r = Sim2RealConfig()
    robot = DoosanRobot()

    print("\n=== Sim2Real Calibration ===\n")

    # Step 1: Home position 비교
    print("Step 1: 로봇을 home position으로 이동하세요")
    input("[Enter] when robot is at home position...")
    real_home, _ = robot.get_state()
    sim_home = np.deg2rad([0, 0, -90, 0, 90, 0])

    offset = real_home - sim_home
    print(f"  Sim home (deg): [{', '.join(f'{d:.1f}' for d in np.rad2deg(sim_home))}]")
    print(f"  Real home (deg): [{', '.join(f'{d:.1f}' for d in np.rad2deg(real_home))}]")
    print(f"  Offset (deg): [{', '.join(f'{d:.1f}' for d in np.rad2deg(offset))}]")

    s2r.joint_offset = offset

    # Step 2: 작은 동작으로 scale 추정
    print("\nStep 2: 각 joint를 10° 움직여보고 실제 변화량을 측정합니다")
    test_delta = np.deg2rad(10)

    for j in range(NUM_JOINTS):
        input(f"  Joint {j+1}: teach pendant로 +10° 이동 후 [Enter]...")
        new_pos, _ = robot.get_state()
        actual_delta = abs(new_pos[j] - real_home[j])
        if actual_delta > 0.001:
            s2r.action_scale[j] = test_delta / actual_delta
        print(f"  Joint {j+1}: 목표=10.0°, 실측={np.rad2deg(actual_delta):.1f}°, "
              f"scale={s2r.action_scale[j]:.3f}")

        # 원위치 복귀
        input(f"  Joint {j+1}을 원래 위치로 복귀 후 [Enter]...")

    # Step 3: 카메라 밝기 비교
    print("\nStep 3: 카메라 이미지 밝기 확인")
    camera = CameraCapture()
    img = camera.read()
    real_brightness = img.mean()
    sim_brightness = 128.0  # sim 평균 밝기 (대략)
    s2r.image_brightness_offset = int(sim_brightness - real_brightness)
    print(f"  Real brightness: {real_brightness:.0f}")
    print(f"  Brightness offset: {s2r.image_brightness_offset}")
    camera.release()

    # 저장
    config_path = os.path.join(
        os.path.dirname(args.checkpoint) if args.checkpoint else ".",
        "sim2real_config.json"
    )
    s2r.save(config_path)

    print(f"\n✅ Calibration 완료")
    print(f"  Action scale: [{', '.join(f'{s:.3f}' for s in s2r.action_scale)}]")
    print(f"  Joint offset (deg): [{', '.join(f'{d:.1f}' for d in np.rad2deg(s2r.joint_offset))}]")
    print(f"  저장: {config_path}")


# =============================================================
# Main
# =============================================================

def main():
    p = argparse.ArgumentParser(description="Sim-to-Real 배포")
    p.add_argument("--mode", choices=["real", "sim-eval", "calibrate"], required=True,
                    help="real: 실제 로봇 배포, sim-eval: sim에서 평가, calibrate: 보정")
    p.add_argument("--checkpoint", default="./checkpoints/groot/checkpoint-10000",
                    help="학습된 모델 checkpoint 경로")
    p.add_argument("--vla-url", default="http://localhost:5555",
                    help="VLA inference server URL")
    p.add_argument("--instruction",
                    default="Pick up the blue medicine bottle and place it in dispensing slot 1")
    p.add_argument("--scenario", default="basic_single",
                    help="sim-eval 모드에서 사용할 시나리오")
    p.add_argument("--hz", type=float, default=float(CONTROL_HZ))
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--execute-horizon", type=int, default=4)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--decay", type=float, default=0.7)
    p.add_argument("--headless", action="store_true")
    # STT
    p.add_argument("--stt", action="store_true")
    p.add_argument("--stt-llm", action="store_true")
    p.add_argument("--stt-duration", type=float, default=5)
    args = p.parse_args()

    if args.mode == "calibrate":
        calibrate(args)
        return

    # STT instruction
    instruction = args.instruction
    if args.stt:
        from utils.stt_instruction import STTInstruction
        stt = STTInstruction(use_llm=args.stt_llm)
        heard = stt.listen(duration=args.stt_duration)
        if heard:
            instruction = heard

    controller = Sim2RealController(args)

    if args.mode == "real":
        controller.run_real(instruction)
    elif args.mode == "sim-eval":
        scenario = get_scenario(args.scenario)
        controller.run_sim_eval(scenario)


if __name__ == "__main__":
    main()
