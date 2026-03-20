"""
약국 조제 보조 시뮬레이션 — 로봇 없이 전체 파이프라인 검증

Mock VLA가 시나리오에 맞는 action chunk를 생성하고,
TemporalBlender → ActionAdapter → FailureDetector 전체 루프를 돌린다.
STT fallback (텍스트 입력)도 테스트 가능.

사용법:
  python utils/pharmacy_sim.py                          # 기본 시나리오
  python utils/pharmacy_sim.py --scenario basic_red     # 빨간 약병
  python utils/pharmacy_sim.py --scenario all           # 전체 시나리오 순회
  python utils/pharmacy_sim.py --inject-failure stall   # stall 실패 주입
  python utils/pharmacy_sim.py --inject-failure clamp   # over-clamp 실패 주입
  python utils/pharmacy_sim.py --stt                    # STT 모드 (텍스트 입력)
"""

import os, sys, time, argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import *
from configs.pharmacy_scenario import (
    MEDICINES, DISPENSING_SLOTS, SCENARIOS,
    SIM_HOME_POSITION, SIM_PRE_GRASP_OFFSET, SIM_LIFT_OFFSET,
    SIM_GRASP_STEPS, SIM_TRANSFER_STEPS, SIM_PLACE_STEPS, SIM_NOISE_SCALE,
    get_scenario,
)
from utils.doosan_action_adapter import DoosanActionAdapter
from utils.doosan_vla_controller import TemporalBlender
from utils.failure_detector import FailureDetector


class MockVLA:
    """시나리오 기반 Mock VLA — 사전 정의된 궤적을 action chunk로 반환"""

    def __init__(self, scenario: dict, chunk_size=16, noise_scale=SIM_NOISE_SCALE):
        self.chunk_size = chunk_size
        self.noise_scale = noise_scale
        self.call_count = 0

        pick_obj = MEDICINES[scenario["pick"]]
        place_slot = DISPENSING_SLOTS[scenario["place"]]

        # 전체 궤적: home → approach → pick → grasp → lift → transfer → place → release → home
        self._trajectory = self._build_trajectory(
            pick_pos=pick_obj["sim_position"],
            place_pos=place_slot["sim_position"],
        )
        self._total_steps = len(self._trajectory)
        self._cursor = 0

    def _build_trajectory(self, pick_pos, place_pos):
        """전체 궤적을 delta action 시퀀스로 생성"""
        waypoints = []
        current = SIM_HOME_POSITION.copy()

        # Phase 1: Home → Pre-grasp
        pre_grasp = pick_pos + SIM_PRE_GRASP_OFFSET
        waypoints.extend(self._interpolate(current, pre_grasp, SIM_GRASP_STEPS, grip=1.0))
        current = pre_grasp

        # Phase 2: Pre-grasp → Pick (그리퍼 열린 채로 접근)
        waypoints.extend(self._interpolate(current, pick_pos, 4, grip=1.0))
        current = pick_pos

        # Phase 3: Grasp (그리퍼 닫기)
        waypoints.extend([(np.zeros(NUM_JOINTS), 0.0)] * 3)  # 정지 + grasp

        # Phase 4: Lift
        lift_pos = pick_pos + SIM_LIFT_OFFSET
        waypoints.extend(self._interpolate(current, lift_pos, 4, grip=0.0))
        current = lift_pos

        # Phase 5: Transfer → Place 위 (그리퍼 닫은 채로 이동)
        pre_place = place_pos + SIM_LIFT_OFFSET
        waypoints.extend(self._interpolate(current, pre_place, SIM_TRANSFER_STEPS, grip=0.0))
        current = pre_place

        # Phase 6: Place (내려놓기)
        waypoints.extend(self._interpolate(current, place_pos, SIM_PLACE_STEPS, grip=0.0))
        current = place_pos

        # Phase 7: Release (그리퍼 열기)
        waypoints.extend([(np.zeros(NUM_JOINTS), 1.0)] * 3)

        # Phase 8: Retreat → Home
        waypoints.extend(self._interpolate(current, SIM_HOME_POSITION, 8, grip=1.0))

        return waypoints

    def _interpolate(self, start, end, steps, grip):
        """두 joint pose 사이를 보간하여 delta action 생성"""
        delta_total = end - start
        actions = []
        for i in range(steps):
            delta = delta_total / steps
            actions.append((delta, grip))
        return actions

    def predict(self, image, state, instruction, execute_horizon=4):
        """Mock VLA prediction — action chunk 반환

        VLA는 매번 chunk_size 길이의 chunk를 출력하지만,
        실제로는 execute_horizon만큼만 실행하고 다시 inference한다.
        따라서 커서를 execute_horizon만큼만 전진시켜 overlap 효과를 시뮬레이션.
        """
        self.call_count += 1
        chunk = []
        for i in range(self.chunk_size):
            idx = self._cursor + i
            if idx < self._total_steps:
                delta, grip = self._trajectory[idx]
                noise = np.random.randn(NUM_JOINTS) * self.noise_scale
                action = np.concatenate([delta + noise, [grip]])
            else:
                action = np.zeros(ACTION_DIM)
            chunk.append(action)

        self._cursor += execute_horizon
        return np.array(chunk)

    @property
    def done(self):
        return self._cursor >= self._total_steps


class MockVLAWithFailure(MockVLA):
    """실패 주입 가능한 Mock VLA"""

    def __init__(self, scenario, failure_type=None, failure_at_step=20, **kwargs):
        super().__init__(scenario, **kwargs)
        self.failure_type = failure_type
        self.failure_at_step = failure_at_step
        self._step_counter = 0

    def predict(self, image, state, instruction, execute_horizon=4):
        chunk = super().predict(image, state, instruction, execute_horizon=execute_horizon)

        if self.failure_type and self._step_counter >= self.failure_at_step:
            if self.failure_type == "stall":
                # 모든 action을 0으로 → stall 감지 유도
                chunk[:, :NUM_JOINTS] = 0.0
            elif self.failure_type == "clamp":
                # 비현실적으로 큰 action → over-clamp 유도
                chunk[:, :NUM_JOINTS] = 0.5

        self._step_counter += self.chunk_size
        return chunk


class SimRobot:
    """시뮬레이션 로봇 — joint state를 추적"""

    def __init__(self):
        self.joints = SIM_HOME_POSITION.copy()
        self.gripper = 0.0
        self.history = []

    def get_state(self):
        return self.joints.copy(), self.gripper

    def send(self, joint_targets, gripper_open, dt=0.1):
        self.joints = joint_targets.copy()
        self.gripper = 1.0 if gripper_open else 0.0
        self.history.append({
            "joints_deg": np.rad2deg(self.joints).tolist(),
            "gripper": "OPEN" if gripper_open else "CLOSE",
        })


def run_simulation(scenario, args):
    """단일 시나리오 시뮬레이션 실행"""
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario['id']} — {scenario['description']}")
    print(f"  Instruction: {scenario['instruction']}")
    print(f"  Pick: {scenario['pick']} → Slot: {scenario['place']}")
    print(f"{'='*60}\n")

    # 컴포넌트 초기화
    if args.inject_failure:
        vla = MockVLAWithFailure(scenario, failure_type=args.inject_failure)
    else:
        vla = MockVLA(scenario)

    robot = SimRobot()
    adapter = DoosanActionAdapter()
    blender = TemporalBlender(
        execute_horizon=args.execute_horizon,
        overlap=args.overlap,
        decay=args.decay,
    )
    detector = FailureDetector()
    dt = 1.0 / args.hz

    instruction = scenario["instruction"]
    if args.stt:
        user_input = input(f"[STT Sim] instruction 입력 (Enter=기본값): ").strip()
        if user_input:
            instruction = user_input

    # 시뮬레이션 루프
    step = 0
    max_steps = args.max_steps
    inference_count = 0
    retry_events = []
    fallback_triggered = False

    while step < max_steps:
        joints, grip = robot.get_state()
        adapter.set_current_state(joints, grip)

        # Mock 이미지 (실제로는 사용 안 함)
        mock_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        chunk = vla.predict(mock_image, np.concatenate([joints, [grip]]), instruction,
                            execute_horizon=args.execute_horizon)
        inference_count += 1

        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        actions = blender.blend(chunk)

        chunk_interrupted = False
        for act in actions:
            if step >= max_steps:
                break

            joints, grip = robot.get_state()
            adapter.set_current_state(joints, grip)
            cmd = adapter.convert(act, dt)
            robot.send(cmd["joint_targets"], cmd["gripper_open"], dt)

            # 실패 감지
            status = detector.update(cmd["joint_targets"], cmd["clamp_ratio"])

            if status["should_fallback"]:
                print(f"  ⚠️  [step {step}] 반복 실패 — classical fallback 전환")
                fallback_triggered = True
                break

            if status["should_retry"]:
                reason = "stall" if status["stalled"] else "over-clamp"
                print(f"  🔄 [step {step}] {reason} 감지 — 재시도 #{detector.retry_count}")
                retry_events.append({"step": step, "reason": reason})
                blender.reset()
                chunk_interrupted = True
                break

            if step % 10 == 0:
                grip_str = "OPEN" if cmd["gripper_open"] else "CLOSE"
                joints_str = ", ".join(f"{d:.1f}" for d in np.rad2deg(cmd["joint_targets"]))
                print(f"  [step {step:3d}] [{joints_str}] {grip_str}  clamp={cmd['clamp_ratio']:.0%}")

            step += 1

        if fallback_triggered:
            break

        if vla.done and not chunk_interrupted:
            print(f"\n  ✅ 궤적 완료 (step {step})")
            break

    # 결과 요약 — 궤적 중 place 위치에 가장 근접한 시점으로 판정
    target_place = np.rad2deg(DISPENSING_SLOTS[scenario["place"]]["sim_position"])
    min_error = float("inf")
    for h in robot.history:
        err = np.abs(np.array(h["joints_deg"]) - target_place).max()
        if err < min_error:
            min_error = err
    final_joints = np.rad2deg(robot.joints)

    print(f"\n{'─'*60}")
    print(f"  결과 요약")
    print(f"{'─'*60}")
    print(f"  총 스텝: {step}")
    print(f"  VLA inference 횟수: {inference_count}")
    print(f"  재시도: {len(retry_events)}회")
    print(f"  Fallback: {'Yes' if fallback_triggered else 'No'}")
    print(f"  최종 joint (deg): [{', '.join(f'{d:.1f}' for d in final_joints)}]")
    print(f"  목표 joint (deg): [{', '.join(f'{d:.1f}' for d in target_place)}]")
    print(f"  최소 위치 오차 (궤적 중): {min_error:.1f}°")
    print(f"  clamp 비율: {adapter.clamp_count}/{adapter.total_count} = {adapter.clamp_count/max(adapter.total_count,1):.0%}")

    success = min_error < 10.0 and not fallback_triggered
    print(f"  판정: {'✅ SUCCESS' if success else '❌ FAIL'}")
    print(f"{'─'*60}")

    return {
        "scenario": scenario["id"],
        "success": success,
        "steps": step,
        "inferences": inference_count,
        "retries": len(retry_events),
        "fallback": fallback_triggered,
        "position_error": min_error,
    }


def main():
    p = argparse.ArgumentParser(description="약국 조제 보조 시뮬레이션")
    p.add_argument("--scenario", default="basic_single",
                    help="시나리오 ID 또는 'all'로 전체 실행")
    p.add_argument("--hz", type=float, default=float(CONTROL_HZ))
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--execute-horizon", type=int, default=4)
    p.add_argument("--overlap", type=int, default=4)
    p.add_argument("--decay", type=float, default=0.7)
    p.add_argument("--inject-failure", choices=["stall", "clamp"], default=None,
                    help="실패 주입: stall 또는 clamp")
    p.add_argument("--stt", action="store_true", help="STT 시뮬레이션 (텍스트 입력)")
    args = p.parse_args()

    if args.scenario == "all":
        results = []
        for scenario in SCENARIOS:
            result = run_simulation(scenario, args)
            results.append(result)

        print(f"\n{'='*60}")
        print(f"  전체 결과")
        print(f"{'='*60}")
        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"  {status} {r['scenario']:20s}  steps={r['steps']:3d}  "
                  f"err={r['position_error']:.1f}°  retries={r['retries']}")
        success_count = sum(1 for r in results if r["success"])
        print(f"\n  성공률: {success_count}/{len(results)}")
    else:
        scenario = get_scenario(args.scenario)
        run_simulation(scenario, args)


if __name__ == "__main__":
    main()
