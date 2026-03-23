from __future__ import annotations

import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType

print("SCRIPT START")

ROBOT_YAML = "/home/danielc174/projects/Robot-Hackathon/assets/curobo/m0609_with_gripper.yaml"


def main():
    print("ENTER MAIN")
    tensor_args = TensorDeviceType(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    cfg = CudaRobotModelConfig.from_robot_yaml_file(
        ROBOT_YAML,
        tensor_args=tensor_args,
    )
    robot = CudaRobotModel(cfg)

    dof = robot.get_dof()
    print("Loaded robot OK")
    print("DOF:", dof)

    q = torch.zeros((1, dof), device=tensor_args.device, dtype=tensor_args.dtype)
    state = robot.get_state(q)

    print("EE position:", state.ee_position)
    print("EE quaternion:", state.ee_quaternion)

    try:
        limits = robot.get_joint_limits()
        print("Joint limits loaded OK")
        print("limits.position.shape:", limits.position.shape)
        print("limits.position:", limits.position)

        if limits.position.ndim == 3:
            print("Lower:", limits.position[0, :, 0])
            print("Upper:", limits.position[0, :, 1])
        elif limits.position.ndim == 2:
            print("Lower:", limits.position[0, :])
            print("Upper:", limits.position[1, :])
        else:
            print("Unexpected joint limit tensor shape")
    except Exception as e:
        print("Joint limit read failed:", e)


if __name__ == "__main__":
    main()