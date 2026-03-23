from __future__ import annotations

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

ROBOT_YAML = "/home/danielc174/projects/Robot-Hackathon/assets/curobo/m0609_with_gripper.yaml"


def main():
    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    cfg_file = load_yaml(ROBOT_YAML)
    urdf_file = cfg_file["robot_cfg"]["kinematics"]["urdf_path"]
    base_link = cfg_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = cfg_file["robot_cfg"]["kinematics"]["ee_link"]

    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=32,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_cfg)

    q_seed = torch.tensor(
        [[1.5708, 0.0, 2.0944, 3.1416, -1.0472, 0.0]],
        device=tensor_args.device,
        dtype=tensor_args.dtype,
    )

    kin_state = ik_solver.fk(q_seed)
    print("seed ee pos:", kin_state.ee_position)
    print("seed ee quat:", kin_state.ee_quaternion)

    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    result = ik_solver.solve_batch(goal)
    print("success:", result.success)
    print("solution:", result.solution)
    print("position error:", result.position_error)
    print("rotation error:", result.rotation_error)


if __name__ == "__main__":
    main()