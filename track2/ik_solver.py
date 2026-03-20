"""
IK Solver: 3D 좌표 → joint angles (degree)
- 우선: cuRobo (collision-aware, GPU 가속)
- 대체: ikpy (순수 Python, 확실히 동작)
"""

import numpy as np

# ===== cuRobo =====
CUROBO_AVAILABLE = False
try:
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    CUROBO_AVAILABLE = True
except ImportError:
    pass

# ===== ikpy =====
IKPY_AVAILABLE = False
try:
    import ikpy.chain
    IKPY_AVAILABLE = True
except ImportError:
    pass


class IKSolverWrapper:
    """통합 IK 인터페이스"""

    def __init__(self, urdf_path):
        self.backend = None
        self.solver = None

        if CUROBO_AVAILABLE:
            try:
                robot_cfg = RobotConfig.from_basic(
                    urdf_path=urdf_path,
                    base_link="base_link",
                    ee_link="gripper_rh_p12_rn_base",
                    tensor_args={"device": "cpu"},
                )
                ik_cfg = IKSolverConfig.load_from_robot_config(robot_cfg, num_seeds=20)
                self.solver = IKSolver(ik_cfg)
                self.backend = "curobo"
                print("[IK] cuRobo initialized (collision-aware)")
            except Exception as e:
                print(f"[IK] cuRobo init failed: {e}")

        if self.backend is None and IKPY_AVAILABLE:
            try:
                self.solver = ikpy.chain.Chain.from_urdf_file(
                    urdf_path,
                    active_links_mask=[False, True, True, True, True, True, True, False],
                )
                self.backend = "ikpy"
                print("[IK] ikpy initialized (no collision check)")
            except Exception as e:
                print(f"[IK] ikpy init failed: {e}")

        if self.backend is None:
            raise RuntimeError(
                "No IK solver available! Install: pip install ikpy  or  pip install curobo"
            )

    def solve(self, target_xyz_mm, orientation_down=True):
        """
        Args:
            target_xyz_mm: [x, y, z] 로봇 베이스 프레임 (mm)
            orientation_down: True면 그리퍼가 아래를 향하는 자세
        Returns:
            [j1, j2, j3, j4, j5, j6] in degrees, 또는 None
        """
        if self.backend == "curobo":
            return self._solve_curobo(target_xyz_mm)
        elif self.backend == "ikpy":
            return self._solve_ikpy(target_xyz_mm)
        return None

    def _solve_curobo(self, target_xyz_mm):
        from curobo.types.math import Pose
        import torch

        pos = torch.tensor(
            [target_xyz_mm[0] / 1000, target_xyz_mm[1] / 1000, target_xyz_mm[2] / 1000],
            dtype=torch.float32,
        )
        # 아래를 향하는 quaternion (w, x, y, z)
        quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        goal = Pose(position=pos.unsqueeze(0), quaternion=quat.unsqueeze(0))
        result = self.solver.solve_single(goal)

        if result.success.item():
            joints_rad = result.solution.cpu().numpy().flatten()
            return np.rad2deg(joints_rad).tolist()
        print("[IK-cuRobo] No solution")
        return None

    def _solve_ikpy(self, target_xyz_mm):
        target_m = np.array(target_xyz_mm) / 1000.0
        try:
            ik = self.solver.inverse_kinematics(target_position=target_m)
            joints_rad = ik[1:7]
            return np.rad2deg(joints_rad).tolist()
        except Exception as e:
            print(f"[IK-ikpy] Failed: {e}")
            return None


if __name__ == "__main__":
    print(f"cuRobo: {CUROBO_AVAILABLE}, ikpy: {IKPY_AVAILABLE}")
    if IKPY_AVAILABLE or CUROBO_AVAILABLE:
        print("At least one IK solver available ✅")
    else:
        print("❌ Install: pip install ikpy")
