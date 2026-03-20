"""
Isaac Sim 데모 수집기 — Domain Randomization + LeRobot v2 형식 저장

scripted policy로 자동 데모를 수집하고,
DR(조명, 위치, 색상 변동)을 적용하여 sim 데모 수백~수천 개를 자동 생성.

사용법:
  python sim/isaac_data_collector.py --num-episodes 200 --headless
  python sim/isaac_data_collector.py --scenario basic_single --num-episodes 50
  python sim/isaac_data_collector.py --scenario all --num-episodes 500 --headless --domain-rand
"""

import os, sys, time, json, argparse
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import (
    NUM_JOINTS, ACTION_DIM, CONTROL_HZ,
    CAMERA_WIDTH, CAMERA_HEIGHT,
)
from configs.pharmacy_scenario import (
    MEDICINES, DISPENSING_SLOTS, SCENARIOS,
    SIM_HOME_POSITION,
    get_scenario,
)


class ScriptedPolicy:
    """Isaac Sim 내 Cartesian 좌표 기반 scripted pick-and-place 정책

    Isaac Sim의 실제 물체 위치를 읽어서 궤적을 계산.
    (pharmacy_sim.py의 joint-space MockVLA와 달리, 여기서는 물리 기반)
    """

    def __init__(self, env, scenario):
        self.env = env
        self.scenario = scenario
        self._phase = "idle"
        self._phase_step = 0
        self._trajectory = []
        self._traj_idx = 0

    def reset(self):
        """에피소드 시작 시 궤적 생성"""
        pick_name = self.scenario["pick"]
        place_slot = self.scenario["place"]

        # Isaac Sim에서 실제 물체 위치 조회
        pick_pos, _ = self.env.get_object_pose(pick_name)
        slot_pos = np.array(self.env.SLOT_POSITIONS[place_slot])

        if pick_pos is None:
            pick_pos = np.array(self.env.MEDICINE_POSITIONS[pick_name])

        # Cartesian waypoint 궤적 정의
        approach_height = 0.15  # 물체 위 15cm
        grasp_height = 0.02    # 물체 살짝 위
        lift_height = 0.20     # 들어올린 높이

        self._trajectory = [
            # (target_xyz, gripper_open, steps, description)
            (np.array([pick_pos[0], pick_pos[1], pick_pos[2] + approach_height]),
             True, 30, "approach"),
            (np.array([pick_pos[0], pick_pos[1], pick_pos[2] + grasp_height]),
             True, 20, "descend"),
            (np.array([pick_pos[0], pick_pos[1], pick_pos[2] + grasp_height]),
             False, 10, "grasp"),
            (np.array([pick_pos[0], pick_pos[1], pick_pos[2] + lift_height]),
             False, 20, "lift"),
            (np.array([slot_pos[0], slot_pos[1], slot_pos[2] + lift_height]),
             False, 40, "transfer"),
            (np.array([slot_pos[0], slot_pos[1], slot_pos[2] + grasp_height]),
             False, 20, "lower"),
            (np.array([slot_pos[0], slot_pos[1], slot_pos[2] + grasp_height]),
             True, 10, "release"),
            (np.array([slot_pos[0], slot_pos[1], slot_pos[2] + approach_height]),
             True, 15, "retreat"),
        ]
        self._traj_idx = 0
        self._phase_step = 0

    def get_action(self, obs):
        """현재 관측에서 action 반환

        Returns:
            action: (ACTION_DIM,) — joint delta + gripper
            done: bool
        """
        if self._traj_idx >= len(self._trajectory):
            return np.zeros(ACTION_DIM), True

        target_xyz, grip_open, total_steps, phase = self._trajectory[self._traj_idx]

        # 현재 end-effector 위치 (간소화: joint state → FK)
        # 실제로는 Isaac Sim의 end-effector prim에서 읽거나 FK 계산
        current_state = obs["state"][:NUM_JOINTS]

        # IK 대신 간소화된 joint-space 보간 사용
        # (실제 배포 시에는 cuRobo IK 또는 Isaac Sim motion_generation 사용)
        home = SIM_HOME_POSITION
        progress = self._phase_step / max(total_steps, 1)

        # 각 phase에 맞는 목표 joint 계산 (heuristic)
        target_joints = self._xyz_to_approx_joints(target_xyz)
        delta = (target_joints - current_state) / max(total_steps - self._phase_step, 1)

        # noise 추가 (학습 일반화용)
        delta += np.random.randn(NUM_JOINTS) * 0.002

        # action 구성
        grip_val = 1.0 if grip_open else 0.0
        action = np.concatenate([delta, [grip_val]])

        self._phase_step += 1
        if self._phase_step >= total_steps:
            self._traj_idx += 1
            self._phase_step = 0

        return action, False

    def _xyz_to_approx_joints(self, xyz):
        """XYZ → 근사 joint angles (단순 heuristic)

        실제 사용 시에는 cuRobo IK 또는 Isaac Sim ArticulationKinematicsSolver 사용.
        여기서는 데이터 수집 구조만 보여주는 placeholder.
        """
        x, y, z = xyz
        j1 = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        j2 = np.deg2rad(-30) + (z - 0.75) * 0.5
        j3 = np.deg2rad(-60) + (r - 0.3) * 1.0
        j4 = 0.0
        j5 = np.deg2rad(90) - j2 - j3
        j6 = 0.0
        return np.array([j1, j2, j3, j4, j5, j6])

    @property
    def current_phase(self):
        if self._traj_idx < len(self._trajectory):
            return self._trajectory[self._traj_idx][3]
        return "done"


class IsaacDataCollector:
    """Isaac Sim 데모 수집 + LeRobot v2 형식 저장"""

    def __init__(self, save_dir, headless=False, enable_dr=False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.enable_dr = enable_dr
        self._env = None

    def _get_env(self):
        """Lazy 환경 생성 (import 시간 절약)"""
        if self._env is None:
            from sim.pharmacy_isaac_env import PharmacyIsaacEnv
            self._env = PharmacyIsaacEnv(
                headless=self.headless,
                enable_dr=self.enable_dr,
            )
        return self._env

    def collect_episode(self, scenario, episode_id):
        """단일 에피소드 수집"""
        env = self._get_env()
        policy = ScriptedPolicy(env, scenario)

        # Reset
        obs = env.reset()
        policy.reset()

        # 수집
        states, actions, images = [], [], []
        instruction = scenario["instruction"]

        step = 0
        max_steps = 300  # 안전 상한

        while step < max_steps:
            action, done = policy.get_action(obs)

            states.append(obs["state"].copy())
            actions.append(action.copy())
            images.append(obs["image"].copy())

            obs, reward, env_done, info = env.step(action)
            step += 1

            if done:
                # 마지막 프레임 추가
                states.append(obs["state"].copy())
                actions.append(np.zeros(ACTION_DIM))
                images.append(obs["image"].copy())
                break

        # 성공 판정
        success = env.is_object_in_slot(scenario["pick"], scenario["place"])

        # 저장
        ep_dir = self.save_dir / f"episode_{episode_id:04d}"
        ep_dir.mkdir(exist_ok=True)

        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.float32)
        images_arr = np.array(images, dtype=np.uint8)

        np.savez_compressed(str(ep_dir / "data.npz"),
                            states=states_arr, actions=actions_arr)
        np.savez_compressed(str(ep_dir / "images.npz"), images=images_arr)

        meta = {
            "episode_id": episode_id,
            "task": instruction,
            "scenario_id": scenario["id"],
            "num_frames": len(states),
            "fps": CONTROL_HZ,
            "success": success,
            "source": "isaac_sim",
            "domain_randomization": self.enable_dr,
        }
        with open(str(ep_dir / "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def collect_batch(self, scenarios, num_episodes):
        """배치 수집 — 시나리오를 순환하며 수집"""
        results = []
        ep_id = 0

        for i in range(num_episodes):
            scenario = scenarios[i % len(scenarios)]
            print(f"\n[{ep_id+1}/{num_episodes}] {scenario['id']}: {scenario['instruction'][:50]}...")

            try:
                meta = self.collect_episode(scenario, ep_id)
                status = "✅" if meta["success"] else "❌"
                print(f"  {status} {meta['num_frames']} frames")
                results.append(meta)
                ep_id += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")

        # 요약
        success_count = sum(1 for r in results if r["success"])
        print(f"\n{'='*50}")
        print(f"  수집 완료: {len(results)}/{num_episodes} episodes")
        print(f"  성공률: {success_count}/{len(results)}")
        print(f"  저장 위치: {self.save_dir}")
        print(f"{'='*50}")

        # 전체 메타데이터 저장
        summary = {
            "total_episodes": len(results),
            "success_rate": success_count / max(len(results), 1),
            "scenarios": list(set(r["scenario_id"] for r in results)),
            "domain_randomization": self.enable_dr,
        }
        with open(str(self.save_dir / "collection_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return results

    def close(self):
        if self._env is not None:
            self._env.close()


def main():
    p = argparse.ArgumentParser(description="Isaac Sim 데모 수집기")
    p.add_argument("--save-dir", default="./data/sim_raw")
    p.add_argument("--scenario", default="all",
                    help="시나리오 ID 또는 'all'")
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--domain-rand", action="store_true")
    args = p.parse_args()

    if args.scenario == "all":
        scenarios = SCENARIOS
    else:
        scenarios = [get_scenario(args.scenario)]

    collector = IsaacDataCollector(
        save_dir=args.save_dir,
        headless=args.headless,
        enable_dr=args.domain_rand,
    )

    try:
        collector.collect_batch(scenarios, args.num_episodes)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
