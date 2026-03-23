from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

M0609_USD = "/home/danielc174/projects/Robot-Hackathon/assets/m0609_with_gripper.usd"

# -----------------------------
# fixed scene layout
# -----------------------------
TABLE_SIZE = (1.28, 1.44, 0.05)
TABLE_CENTER_POS = (0.0, 0.0, 1.0)
TABLE_TOP_Z = TABLE_CENTER_POS[2] + TABLE_SIZE[2] / 2.0

ROBOT_BASE_POS = (0.0, -0.61, TABLE_TOP_Z)

# -----------------------------
# cube / target
# -----------------------------
CUBE_SIZE = (0.04, 0.04, 0.04)
CUBE_DEFAULT_POS = (0.30, 0.0, TABLE_TOP_Z + CUBE_SIZE[2] / 2.0)

CUBE_X_RANGE = (-0.5, 0.5)
CUBE_Y_RANGE = (-0.20, 0.40)
CUBE_Z = TABLE_TOP_Z + CUBE_SIZE[2] / 2.0

TARGET_SIZE = (0.10, 0.10, 0.01)
TARGET_POS = (0.0, 0.45, TABLE_TOP_Z + TARGET_SIZE[2] / 2.0)

TARGET_X_RANGE = (-0.30, 0.30)
TARGET_Y_RANGE = (0.20, 0.55)
TARGET_Z = TABLE_TOP_Z + TARGET_SIZE[2] / 2.0


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TABLE_CENTER_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=TARGET_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 0.1),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUBE_DEFAULT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        articulation_root_prim_path="/m0609",
        spawn=sim_utils.UsdFileCfg(
            usd_path=M0609_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # chosen downward-facing home pose
                "joint_1": 1.5708,
                "joint_2": 0.0000,
                "joint_3": 2.0944,
                "joint_4": 3.1416,
                "joint_5": -1.0472,
                "joint_6": 0.0000,
                "finger_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "left_inner_finger_knuckle_joint": 0.0,
                "right_inner_finger_knuckle_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit_sim=400.0,
                velocity_limit_sim=10.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper_main": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=400.0,
                damping=20.0,
            ),
            "gripper_mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_outer_knuckle_joint",
                    "left_inner_finger_joint",
                    "right_inner_finger_joint",
                    "left_inner_finger_knuckle_joint",
                    "right_inner_finger_knuckle_joint",
                ],
                effort_limit_sim=10.0,
                velocity_limit_sim=2.0,
                stiffness=50.0,
                damping=5.0,
            ),
        },
    )

    # fixed scene camera for now
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/m0609/link_6/wrist_cam",
        update_period=1.0 / 15.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0032, -0.9505, -0.0288, -0.3093),
            convention="world",
        ),
    )

    def __post_init__(self):
        super().__post_init__()