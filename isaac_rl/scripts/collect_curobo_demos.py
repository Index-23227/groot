# C:\Users\daniel\projects\Hackathon\scripts\collect_curobo_demos.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--horizon", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# imports after app launch
# -----------------------------------------------------------------------------

import torch
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

# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------

ROOT = Path(r"C:\Users\daniel\projects\Hackathon")
DATASET_DIR = ROOT / "data" / "curobo_demos"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# randomization
# -----------------------------------------------------------------------------

def randomize_cube(scene: InteractiveScene, device: str):
    cube = scene["cube"]
    num_envs = scene.num_envs
    env_ids = torch.arange(num_envs, device=device)

    root_state = cube.data.default_root_state.clone()
    root_state[:, 0] = scene.env_origins[:, 0] + torch.empty(num_envs, device=device).uniform_(*CUBE_X_RANGE)
    root_state[:, 1] = scene.env_origins[:, 1] + torch.empty(num_envs, device=device).uniform_(*CUBE_Y_RANGE)
    root_state[:, 2] = scene.env_origins[:, 2] + CUBE_Z
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0
    cube.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_target(scene: InteractiveScene, device: str):
    target = scene["target"]
    num_envs = scene.num_envs
    env_ids = torch.arange(num_envs, device=device)

    root_state = target.data.default_root_state.clone()
    root_state[:, 0] = scene.env_origins[:, 0] + torch.empty(num_envs, device=device).uniform_(*TARGET_X_RANGE)
    root_state[:, 1] = scene.env_origins[:, 1] + torch.empty(num_envs, device=device).uniform_(*TARGET_Y_RANGE)
    root_state[:, 2] = scene.env_origins[:, 2] + TARGET_Z
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0
    target.write_root_state_to_sim(root_state, env_ids=env_ids)


# -----------------------------------------------------------------------------
# state reading
# -----------------------------------------------------------------------------

def get_obs(scene: InteractiveScene, ee_body_idx: int, camera_ready: bool) -> dict:
    robot = scene["robot"]
    camera = scene["camera"]

    obs = {
        "joint_pos":    robot.data.joint_pos.clone(),
        "joint_vel":    robot.data.joint_vel.clone(),
        "ee_pose_w":    robot.data.body_state_w[:, ee_body_idx, :7].clone(),
        "cube_pose_w":  scene["cube"].data.root_state_w[:, :7].clone(),
        "target_pose_w": scene["target"].data.root_state_w[:, :7].clone(),
        "rgb":   camera.data.output["rgb"].clone() if camera_ready and "rgb" in camera.data.output else None,
        "depth": camera.data.output["distance_to_image_plane"].clone()
                 if camera_ready and "distance_to_image_plane" in camera.data.output else None,
    }
    return obs


def get_gripper_state(scene: InteractiveScene, gf_idx: int) -> torch.Tensor:
    robot = scene["robot"]
    return robot.data.joint_pos[:, gf_idx:gf_idx + 1].clone()


# -----------------------------------------------------------------------------
# cuRobo planner stub — replace with real implementation
# -----------------------------------------------------------------------------

def plan_with_curobo(obs: dict, horizon: int, device: str) -> torch.Tensor:
    """
    TODO: replace with real cuRobo planner.
    Inputs:
        obs["joint_pos"]     [num_envs, 12]
        obs["ee_pose_w"]     [num_envs, 7]
        obs["cube_pose_w"]   [num_envs, 7]
        obs["target_pose_w"] [num_envs, 7]
    Returns:
        joint position targets [num_envs, horizon, 6]  (arm joints only)
    """
    num_envs = obs["joint_pos"].shape[0]
    # stub: return current joint pos repeated for all timesteps
    arm_pos = obs["joint_pos"][:, :6].unsqueeze(1).expand(num_envs, horizon, 6)
    return arm_pos.clone()


# -----------------------------------------------------------------------------
# saving
# -----------------------------------------------------------------------------

def save_episode(episode: dict, episode_idx: int):
    save_path = DATASET_DIR / f"episode_{episode_idx:06d}.pt"
    torch.save(episode, save_path)
    print(f"[saved] {save_path}  ({len(episode['steps'])} steps)")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 2.2, 2.2], [0.0, 0.0, TABLE_TOP_Z])

    scene_cfg = ReachSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]
    camera = scene["camera"]

    # resolve indices once
    ee_body_idx = robot.find_bodies("link_6")[0][0]
    gf_idx      = robot.find_joints("finger_joint")[0][0]
    arm_indices = [robot.find_joints(f"joint_{i}")[0][0] for i in range(1, 7)]

    print("=== Diagnostics ===")
    print("  joint names:", robot.joint_names)
    print("  body names :", robot.body_names)
    print("  actuators  :", list(robot.actuators.keys()))
    print("  ee_body_idx:", ee_body_idx)
    print("  arm_indices:", arm_indices)
    print("===================")

    # camera warmup: update_period=1/15s, dt=1/120s → fires every 8 steps
    print("Warming up camera...")
    for _ in range(16):
        sim.step()
        scene.update(sim.get_physics_dt())
    camera_ready = True
    print("Camera ready.")

    home = robot.data.default_joint_pos.clone()

    for ep_idx in range(args_cli.num_episodes):
        print(f"\n[episode {ep_idx}] randomizing scene...")

        # reset robot to home
        robot.write_joint_state_to_sim(home, torch.zeros_like(home))
        scene.write_data_to_sim()

        randomize_cube(scene, device)
        randomize_target(scene, device)

        # step a few times to let physics settle after reset
        for _ in range(8):
            sim.step()
            scene.update(sim.get_physics_dt())

        obs = get_obs(scene, ee_body_idx, camera_ready)

        # record initial poses for episode metadata
        cube_init_pose   = obs["cube_pose_w"].cpu().clone()
        target_init_pose = obs["target_pose_w"].cpu().clone()

        # plan
        planned_actions = plan_with_curobo(obs, args_cli.horizon, device)
        # planned_actions: [num_envs, horizon, 6]

        episode = {
            "steps": [],
            "meta": {
                "episode_idx":     ep_idx,
                "cube_init_pose":  cube_init_pose,
                "target_init_pose": target_init_pose,
            },
        }

        for t in range(args_cli.horizon):
            # build full joint target: arm from planner, gripper from current state
            joint_pos_target = home.clone()
            for i, arm_idx in enumerate(arm_indices):
                joint_pos_target[:, arm_idx] = planned_actions[:, t, i]

            robot.set_joint_position_target(joint_pos_target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

            obs = get_obs(scene, ee_body_idx, camera_ready)
            gripper_state = get_gripper_state(scene, gf_idx)

            # TODO: replace with real termination / success logic
            done    = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
            success = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

            episode["steps"].append({
                "rgb":           obs["rgb"].cpu() if obs["rgb"] is not None else None,
                "depth":         obs["depth"].cpu() if obs["depth"] is not None else None,
                "joint_pos":     obs["joint_pos"].cpu(),
                "joint_vel":     obs["joint_vel"].cpu(),
                "gripper_state": gripper_state.cpu(),
                "ee_pose_w":     obs["ee_pose_w"].cpu(),
                "cube_pose_w":   obs["cube_pose_w"].cpu(),
                "target_pose_w": obs["target_pose_w"].cpu(),
                "action":        planned_actions[:, t, :].cpu(),
                "done":          done.cpu(),
                "success":       success.cpu(),
                "step":          t,
            })

            if done.any():
                print(f"  episode {ep_idx} terminated at step {t}")
                break

        save_episode(episode, ep_idx)

    print(f"\nDone. {args_cli.num_episodes} episodes saved to {DATASET_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()