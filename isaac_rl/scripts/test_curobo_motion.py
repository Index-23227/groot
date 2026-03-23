from __future__ import annotations

import os
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

ROBOT_YAML = "/home/danielc174/projects/Robot-Hackathon/assets/curobo/m0609_with_gripper.yaml"
OUT_PATH = "/home/danielc174/projects/Robot-Hackathon/debug/curobo_traj.pt"


def main():
    os.makedirs("/home/danielc174/projects/Robot-Hackathon/debug", exist_ok=True)

    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    cfg = load_yaml(ROBOT_YAML)["robot_cfg"]
    kin = cfg["kinematics"]
    cspace = cfg["cspace"]

    robot_cfg = RobotConfig.from_basic(
        kin["urdf_path"],
        kin["base_link"],
        kin["ee_link"],
        tensor_args,
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        None,
        tensor_args,
        use_cuda_graph=False,
        self_collision_check=False,
        self_collision_opt=False,
    )
    motion_gen = MotionGen(motion_gen_config)

    print("warming up...")
    motion_gen.warmup(enable_graph=False)
    print("warmup done")

    joint_names = cspace["joint_names"]
    start_q = torch.tensor(
        [cspace["retract_config"]],
        device=tensor_args.device,
        dtype=tensor_args.dtype,
    )
    start_state = JointState.from_position(start_q, joint_names=joint_names)

    kin_state = motion_gen.compute_kinematics(start_state)
    print("start ee pos:", kin_state.ee_pos_seq)
    print("start ee quat:", kin_state.ee_quat_seq)

    goal_pos = kin_state.ee_pos_seq.clone()
    goal_quat = kin_state.ee_quat_seq.clone()
    goal_pos[0, 2] -= 0.05

    goal_pose = Pose(position=goal_pos, quaternion=goal_quat)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_opt=True,
        enable_finetune_trajopt=True,
        max_attempts=1,
        num_trajopt_seeds=8,
        num_graph_seeds=0,
    )

    result = motion_gen.plan_single(start_state, goal_pose, plan_config)
    print("success:", result.success)

    if not bool(result.success.item()):
        raise RuntimeError("planning failed")

    traj = result.get_interpolated_plan().position.detach().cpu()
    print("traj shape:", traj.shape)
    print("first q:", traj[0])
    print("last q:", traj[-1])

    torch.save(traj, OUT_PATH)
    print(f"saved traj to: {OUT_PATH}")


if __name__ == "__main__":
    main()