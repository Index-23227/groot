"""
Isaac Sim 약국 조제 보조 환경

Doosan E0509 + 약품 + 조제함을 Isaac Sim 내에 구성.
카메라 렌더링, 물리 시뮬레이션, Domain Randomization 지원.

요구사항:
  - Isaac Sim 4.0+ (isaacsim pip package 또는 Omniverse Launcher)
  - NVIDIA GPU (RTX 3090 이상 권장)

사용법:
  python sim/pharmacy_isaac_env.py                       # 환경 생성 + 테스트
  python sim/pharmacy_isaac_env.py --headless             # headless 모드
  python sim/pharmacy_isaac_env.py --domain-rand          # Domain Randomization ON
"""

import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.doosan_e0509_config import (
    NUM_JOINTS, ACTION_DIM, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER,
    CONTROL_HZ, CAMERA_WIDTH, CAMERA_HEIGHT,
)
from configs.pharmacy_scenario import (
    MEDICINES, DISPENSING_SLOTS, SCENARIOS,
    SIM_HOME_POSITION,
)


# =============================================================
# Isaac Sim Environment
# =============================================================

class PharmacyIsaacEnv:
    """Isaac Sim 기반 약국 조제 환경

    Scene 구성:
      - Doosan E0509 로봇 (URDF → USD 변환)
      - 약품 4종 (기본 형상 + 색상)
      - 조제함 3슬롯
      - Global camera (45° 상단)
      - 테이블 + 바닥
    """

    # ── 물체 크기/위치 (미터) ──
    TABLE_SIZE = (0.8, 0.6, 0.75)   # 폭, 깊이, 높이
    TABLE_POS  = (0.4, 0.0, 0.375)  # 로봇 앞 40cm

    MEDICINE_POSITIONS = {
        "blue_bottle":  (0.25, -0.15, 0.80),
        "red_bottle":   (0.25,  0.00, 0.80),
        "white_bottle": (0.25,  0.15, 0.80),
        "yellow_box":   (0.20,  0.00, 0.78),
    }

    SLOT_POSITIONS = {
        1: (0.55, -0.12, 0.78),
        2: (0.55,  0.00, 0.78),
        3: (0.55,  0.12, 0.78),
    }

    CAMERA_POS    = (-0.1, 0.0, 1.4)   # 로봇 뒤쪽 상단
    CAMERA_TARGET = (0.4, 0.0, 0.75)    # 테이블 중심을 봄

    def __init__(self, headless=False, enable_dr=False):
        self.headless = headless
        self.enable_dr = enable_dr
        self._step_count = 0
        self._sim_app = None
        self._world = None
        self._robot = None
        self._camera = None
        self._objects = {}

        self._init_sim()
        self._build_scene()
        if enable_dr:
            self._setup_domain_randomization()

    # ── 초기화 ──

    def _init_sim(self):
        """Isaac Sim 앱 초기화"""
        from isaacsim import SimulationApp

        config = {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "headless": self.headless,
            "renderer": "RayTracedLighting",
        }
        self._sim_app = SimulationApp(config)

        from omni.isaac.core import World
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 120.0,        # 물리 120Hz
            rendering_dt=1.0 / CONTROL_HZ,  # 렌더링 = 제어 Hz
        )
        print(f"[Isaac] World created: physics=120Hz, render={CONTROL_HZ}Hz")

    def _build_scene(self):
        """Scene 구성: 로봇, 테이블, 약품, 조제함, 카메라"""
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, DynamicCylinder
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.prims import XFormPrim
        import omni.isaac.core.utils.prims as prim_utils

        # ── 바닥 ──
        self._world.scene.add_default_ground_plane()

        # ── 테이블 ──
        self._world.scene.add(FixedCuboid(
            prim_path="/World/table",
            name="table",
            position=np.array(self.TABLE_POS),
            scale=np.array(self.TABLE_SIZE),
            color=np.array([0.6, 0.5, 0.4]),  # 나무색
        ))

        # ── 로봇 ──
        robot_usd = self._get_robot_usd()
        add_reference_to_stage(usd_path=robot_usd, prim_path="/World/doosan_e0509")
        self._robot = self._world.scene.add(Robot(
            prim_path="/World/doosan_e0509",
            name="doosan_e0509",
            position=np.array([0.0, 0.0, 0.75]),  # 테이블 위
        ))

        # ── 약품 ──
        color_map = {
            "blue_bottle":  [0.1, 0.3, 0.9],
            "red_bottle":   [0.9, 0.1, 0.1],
            "white_bottle": [0.95, 0.95, 0.95],
            "yellow_box":   [0.95, 0.85, 0.1],
        }

        for med_key, pos in self.MEDICINE_POSITIONS.items():
            med = MEDICINES[med_key]
            color = np.array(color_map[med_key])

            if "box" in med_key:
                obj = self._world.scene.add(DynamicCuboid(
                    prim_path=f"/World/{med_key}",
                    name=med_key,
                    position=np.array(pos),
                    scale=np.array([0.05, 0.08, 0.03]),  # 5×8×3cm 상자
                    color=color,
                    mass=0.05,
                ))
            else:
                obj = self._world.scene.add(DynamicCylinder(
                    prim_path=f"/World/{med_key}",
                    name=med_key,
                    position=np.array(pos),
                    radius=0.025,  # 지름 5cm
                    height=0.12,   # 높이 12cm
                    color=color,
                    mass=0.05,
                ))
            self._objects[med_key] = obj

        # ── 조제함 (3슬롯) ──
        slot_colors = [[0.7, 0.7, 0.7], [0.75, 0.75, 0.75], [0.8, 0.8, 0.8]]
        for slot_id, pos in self.SLOT_POSITIONS.items():
            # 바닥판
            self._world.scene.add(FixedCuboid(
                prim_path=f"/World/slot_{slot_id}_base",
                name=f"slot_{slot_id}_base",
                position=np.array(pos),
                scale=np.array([0.08, 0.08, 0.005]),
                color=np.array(slot_colors[slot_id - 1]),
            ))
            # 뒷벽 (약이 넘어가지 않도록)
            self._world.scene.add(FixedCuboid(
                prim_path=f"/World/slot_{slot_id}_wall",
                name=f"slot_{slot_id}_wall",
                position=np.array([pos[0] + 0.04, pos[1], pos[2] + 0.03]),
                scale=np.array([0.005, 0.08, 0.06]),
                color=np.array(slot_colors[slot_id - 1]),
            ))

        # ── 카메라 ──
        from omni.isaac.sensor import Camera
        self._camera = Camera(
            prim_path="/World/camera_global",
            position=np.array(self.CAMERA_POS),
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=CONTROL_HZ,
        )
        self._camera.set_focal_length(1.8)
        # 카메라가 테이블 중심을 보도록
        self._camera.set_world_pose(
            position=np.array(self.CAMERA_POS),
            orientation=self._look_at_quat(self.CAMERA_POS, self.CAMERA_TARGET),
        )
        self._world.scene.add(self._camera)

        # ── 초기화 ──
        self._world.reset()
        self._set_robot_joints(SIM_HOME_POSITION)
        print("[Isaac] Scene built: robot + 4 medicines + 3 slots + camera")

    def _get_robot_usd(self):
        """E0509 USD 경로 반환 (없으면 URDF→USD 변환)"""
        usd_path = os.path.join(os.path.dirname(__file__), "assets", "doosan_e0509.usd")
        if os.path.exists(usd_path):
            return usd_path

        # URDF가 있으면 변환 시도
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "doosan_e0509.urdf")
        if os.path.exists(urdf_path):
            from omni.isaac.urdf import _urdf
            cfg = _urdf.ImportConfig()
            cfg.fix_base = True
            cfg.make_default_prim = True
            result = _urdf.import_robot(urdf_path, usd_path, cfg)
            print(f"[Isaac] URDF → USD 변환 완료: {usd_path}")
            return usd_path

        # 둘 다 없으면 NVIDIA 기본 로봇 사용 (placeholder)
        print("[Isaac] ⚠️  E0509 USD/URDF 없음 — Franka를 placeholder로 사용")
        print("        sim/assets/doosan_e0509.urdf 또는 .usd를 준비해주세요")
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        return f"{assets_root}/Isaac/Robots/Franka/franka_instanceable.usd"

    # ── 제어 인터페이스 ──

    def step(self, action: np.ndarray):
        """환경 1스텝 진행

        Args:
            action: (ACTION_DIM,) — joint delta (6) + gripper (1)

        Returns:
            obs: dict with 'image', 'state'
            reward: float
            done: bool
            info: dict
        """
        assert action.shape == (ACTION_DIM,), f"Expected {ACTION_DIM}-dim, got {action.shape}"

        # Joint delta 적용
        current_joints = self._get_robot_joints()
        target_joints = current_joints + action[:NUM_JOINTS]
        target_joints = np.clip(target_joints, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        self._set_robot_joints(target_joints)

        # Gripper
        gripper_open = action[NUM_JOINTS] > 0.5
        self._set_gripper(gripper_open)

        # 물리 시뮬레이션 진행 (120Hz / CONTROL_HZ = 12스텝)
        physics_steps = int(120.0 / CONTROL_HZ)
        for _ in range(physics_steps):
            self._world.step(render=False)
        self._world.step(render=True)  # 마지막에 렌더링

        self._step_count += 1

        # Observation
        obs = self.get_observation()

        # Reward (간단한 거리 기반)
        reward = 0.0
        done = False

        return obs, reward, done, {"step": self._step_count}

    def get_observation(self):
        """현재 관측 반환"""
        image = self._camera.get_rgba()[:, :, :3]  # RGBA → RGB
        joints = self._get_robot_joints()
        gripper = 1.0 if self._get_gripper_state() else 0.0

        return {
            "image": image,                                    # (H, W, 3) uint8
            "state": np.concatenate([joints, [gripper]]),      # (7,) float
        }

    def reset(self, scenario_id=None):
        """환경 리셋 (선택적으로 특정 시나리오 배치)"""
        self._world.reset()
        self._set_robot_joints(SIM_HOME_POSITION)
        self._step_count = 0

        # 약품 초기 위치로 복원
        for med_key, pos in self.MEDICINE_POSITIONS.items():
            if med_key in self._objects:
                self._objects[med_key].set_world_pose(position=np.array(pos))
                self._objects[med_key].set_linear_velocity(np.zeros(3))
                self._objects[med_key].set_angular_velocity(np.zeros(3))

        if self.enable_dr:
            self._apply_domain_randomization()

        # 물리 안정화 (몇 스텝 진행)
        for _ in range(10):
            self._world.step(render=False)
        self._world.step(render=True)

        return self.get_observation()

    def close(self):
        """환경 종료"""
        if self._sim_app is not None:
            self._sim_app.close()
            print("[Isaac] Closed")

    # ── Domain Randomization ──

    def _setup_domain_randomization(self):
        """DR 파라미터 설정"""
        self._dr_config = {
            "light_intensity_range": (500, 3000),
            "light_color_range": (0.8, 1.0),       # 따뜻~차가운 조명
            "object_pos_noise": 0.03,               # ±3cm 위치 변동
            "object_color_noise": 0.1,              # 색상 약간 변동
            "camera_pos_noise": 0.05,               # ±5cm 카메라 위치
            "table_color_range": [(0.4, 0.3, 0.2), (0.8, 0.7, 0.6)],
        }
        print(f"[Isaac] Domain Randomization ON: {self._dr_config}")

    def _apply_domain_randomization(self):
        """매 에피소드 시작 시 랜덤화 적용"""
        cfg = self._dr_config

        # 약품 위치 랜덤화
        for med_key, base_pos in self.MEDICINE_POSITIONS.items():
            if med_key in self._objects:
                noise = np.random.uniform(-cfg["object_pos_noise"],
                                          cfg["object_pos_noise"], 3)
                noise[2] = 0  # 높이는 유지
                new_pos = np.array(base_pos) + noise
                self._objects[med_key].set_world_pose(position=new_pos)

        # 카메라 위치 약간 랜덤화
        cam_noise = np.random.uniform(-cfg["camera_pos_noise"],
                                       cfg["camera_pos_noise"], 3)
        new_cam_pos = np.array(self.CAMERA_POS) + cam_noise
        self._camera.set_world_pose(
            position=new_cam_pos,
            orientation=self._look_at_quat(new_cam_pos, self.CAMERA_TARGET),
        )

    # ── 로봇 제어 헬퍼 ──

    def _get_robot_joints(self):
        """현재 joint positions (rad)"""
        if self._robot is not None:
            pos = self._robot.get_joint_positions()
            if pos is not None and len(pos) >= NUM_JOINTS:
                return pos[:NUM_JOINTS]
        return SIM_HOME_POSITION.copy()

    def _set_robot_joints(self, positions):
        """Joint positions 설정"""
        if self._robot is not None:
            self._robot.set_joint_positions(positions[:NUM_JOINTS])

    def _set_gripper(self, open_state: bool):
        """그리퍼 제어"""
        if self._robot is not None:
            # E0509 그리퍼 = 마지막 joint 또는 별도 제어
            # USD 모델에 따라 조정 필요
            try:
                if open_state:
                    self._robot.gripper.open()
                else:
                    self._robot.gripper.close()
            except AttributeError:
                pass  # 그리퍼 없는 모델

    def _get_gripper_state(self):
        """그리퍼 상태 반환 (True=open)"""
        try:
            return self._robot.gripper.is_open()
        except AttributeError:
            return True

    # ── 유틸리티 ──

    @staticmethod
    def _look_at_quat(eye, target):
        """eye에서 target을 바라보는 quaternion 계산"""
        import scipy.spatial.transform as tf
        forward = np.array(target) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rot_mat = np.stack([right, up, -forward], axis=1)
        r = tf.Rotation.from_matrix(rot_mat)
        return r.as_quat()[[3, 0, 1, 2]]  # Isaac Sim: w, x, y, z

    def get_object_pose(self, name):
        """물체의 world pose 반환"""
        if name in self._objects:
            pos, quat = self._objects[name].get_world_pose()
            return pos, quat
        return None, None

    def is_object_in_slot(self, obj_name, slot_id, threshold=0.05):
        """물체가 특정 슬롯 안에 있는지 판정"""
        pos, _ = self.get_object_pose(obj_name)
        if pos is None:
            return False
        slot_pos = np.array(self.SLOT_POSITIONS[slot_id])
        dist = np.linalg.norm(pos[:2] - slot_pos[:2])  # XY 평면 거리
        return dist < threshold


# =============================================================
# 테스트
# =============================================================

def test_env(headless=False, domain_rand=False):
    """환경 생성 + 기본 동작 테스트"""
    env = PharmacyIsaacEnv(headless=headless, enable_dr=domain_rand)

    try:
        # Reset
        obs = env.reset()
        print(f"\n[Test] Observation:")
        print(f"  image: {obs['image'].shape} {obs['image'].dtype}")
        print(f"  state: {obs['state'].shape} → [{', '.join(f'{x:.3f}' for x in obs['state'])}]")

        # 몇 스텝 실행
        print("\n[Test] Running 50 steps with random actions...")
        for i in range(50):
            action = np.random.randn(ACTION_DIM) * 0.01
            obs, reward, done, info = env.step(action)
            if i % 10 == 0:
                joints_deg = np.rad2deg(obs["state"][:NUM_JOINTS])
                print(f"  step {i:3d}: [{', '.join(f'{d:.1f}' for d in joints_deg)}]")

        # 물체 위치 확인
        print("\n[Test] Object positions:")
        for med_key in MEDICINES:
            pos, _ = env.get_object_pose(med_key)
            if pos is not None:
                print(f"  {med_key}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Slot 판정 테스트
        for slot_id in DISPENSING_SLOTS:
            for med_key in MEDICINES:
                if env.is_object_in_slot(med_key, slot_id):
                    print(f"  ✅ {med_key} is in slot {slot_id}")

        print("\n✅ Environment test passed")

    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Isaac Sim 약국 환경 테스트")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--domain-rand", action="store_true")
    args = p.parse_args()
    test_env(headless=args.headless, domain_rand=args.domain_rand)
