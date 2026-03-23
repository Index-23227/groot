from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs.reach_scene_cfg import (
    ReachSceneCfg,
    TABLE_TOP_Z,
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    CUBE_Z,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
    TARGET_Z,
)

TRAJ_PATH = "/home/danielc174/projects/Robot-Hackathon/debug/curobo_traj.pt"
IMG_DIR = "/home/danielc174/projects/Robot-Hackathon/debug"


def randomize_cube(scene: InteractiveScene, sim: sim_utils.SimulationContext):
    cube = scene["cube"]
    num_envs = scene.num_envs
    env_ids = torch.arange(num_envs, device=sim.device)

    root_state = cube.data.default_root_state.clone()
    rand_x = torch.empty(num_envs, device=sim.device).uniform_(*CUBE_X_RANGE)
    rand_y = torch.empty(num_envs, device=sim.device).uniform_(*CUBE_Y_RANGE)

    root_state[:, 0] = scene.env_origins[:, 0] + rand_x
    root_state[:, 1] = scene.env_origins[:, 1] + rand_y
    root_state[:, 2] = scene.env_origins[:, 2] + CUBE_Z
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0

    cube.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_target(scene: InteractiveScene, sim: sim_utils.SimulationContext):
    target = scene["target"]
    num_envs = scene.num_envs
    env_ids = torch.arange(num_envs, device=sim.device)

    root_state = target.data.default_root_state.clone()
    rand_x = torch.empty(num_envs, device=sim.device).uniform_(*TARGET_X_RANGE)
    rand_y = torch.empty(num_envs, device=sim.device).uniform_(*TARGET_Y_RANGE)

    root_state[:, 0] = scene.env_origins[:, 0] + rand_x
    root_state[:, 1] = scene.env_origins[:, 1] + rand_y
    root_state[:, 2] = scene.env_origins[:, 2] + TARGET_Z
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0

    target.write_root_state_to_sim(root_state, env_ids=env_ids)


def save_one_camera_image(camera, out_dir=IMG_DIR, name="camera_rgb.png"):
    os.makedirs(out_dir, exist_ok=True)
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    path = os.path.join(out_dir, name)
    Image.fromarray(rgb).save(path)
    print(f"Saved RGB image to: {path}")


def main():
    traj = torch.load(TRAJ_PATH)
    print("loaded traj:", traj.shape)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 2.2, 2.2], [0.0, 0.0, TABLE_TOP_Z])

    scene_cfg = ReachSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    randomize_cube(scene, sim)
    randomize_target(scene, sim)

    robot = scene["robot"]
    camera = scene["camera"]

    home = robot.data.default_joint_pos.clone()

    robot.set_joint_position_target(home)
    scene.write_data_to_sim()
    for _ in range(20):
        sim.step()
        scene.update(sim.get_physics_dt())

    arm_joint_ids = [
        robot.find_joints("joint_1")[0][0],
        robot.find_joints("joint_2")[0][0],
        robot.find_joints("joint_3")[0][0],
        robot.find_joints("joint_4")[0][0],
        robot.find_joints("joint_5")[0][0],
        robot.find_joints("joint_6")[0][0],
    ]

    step_count = 0
    saved = False

    while simulation_app.is_running():
        if sim.is_playing():
            traj_idx = min(step_count, traj.shape[0] - 1)

            arm_target = traj[traj_idx].to(device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)

            robot.set_joint_position_target(arm_target, joint_ids=arm_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

            if not saved and step_count >= 5:
                save_one_camera_image(camera)
                saved = True

            step_count += 1

            if step_count >= traj.shape[0] + 30:
                break
        else:
            sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()