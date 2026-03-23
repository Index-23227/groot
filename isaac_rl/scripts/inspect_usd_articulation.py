from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import UsdPhysics

USD_PATH = "/home/danielc174/projects/Robot-Hackathon/assets/m0609_with_gripper.usd"

ctx = omni.usd.get_context()
ok = ctx.open_stage(USD_PATH)
print("open_stage:", ok)

stage = ctx.get_stage()
if stage is None:
    raise RuntimeError(f"Failed to open stage: {USD_PATH}")

print("=== ArticulationRootAPI prims ===")
found = False
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(prim.GetPath())
        found = True
if not found:
    print("NONE")

print("\n=== Top-level children ===")
for child in stage.GetPseudoRoot().GetChildren():
    print(child.GetPath(), child.GetTypeName())

simulation_app.close()