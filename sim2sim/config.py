"""Configuration for the lightweight MuJoCo sim2sim runner."""

import os


SIM2SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SIM2SIM_DIR, "assets")
# TRAIN_RUN_DIR = (
#     "/path/to/unitree_rl_lab/logs/rsl_rl/"
#     "unitree_g1_29dof_velocity_rough/2026-06-12_10-36-30"
# )
TRAIN_RUN_DIR = os.path.join(SIM2SIM_DIR, "policy", "2026-06-12_10-36-30")

# ROBOT_SCENE = os.path.join(ASSETS_DIR, "scene_flat.xml")
# ROBOT_SCENE = os.path.join(ASSETS_DIR, "scene_rough.xml")
ROBOT_SCENE = os.path.join(ASSETS_DIR, "scene_plum_blossom.xml")

RAYCASTER_PLUGIN_LIBRARY = os.path.join(SIM2SIM_DIR, "libsensor_raycaster.so")

POLICY_PATH = os.path.join(os.path.dirname(SIM2SIM_DIR), "pretrained", "ame1.pt")
DEPLOY_CONFIG = os.path.join(SIM2SIM_DIR, "deploy.yaml")

SIM_DT = 0.002
DECIMATION = 10
DEVICE = "cpu"

HEIGHT_SCANNER_SENSOR_NAME = "height_scanner"
HEIGHT_SCANNER_CAMERA_NAME = "height_scanner_camera"
HEIGHT_SCANNER_OFFSET = 0.5
HEIGHT_SCANNER_CLIP = (-1.0, 5.0)
HEIGHT_SCANNER_FLIP_Y_FOR_ISAACLAB_ORDER = True

DRAW_HEIGHT_SCANNER_POINTS = True
HEIGHT_SCANNER_POINT_RADIUS = 0.02
HEIGHT_SCANNER_POINT_RGBA = (1.0, 0.05, 0.05, 1.0)

VIEWER_LOOKAT = (7.0, 0.0, 0.4)
VIEWER_DISTANCE = 25.0
VIEWER_AZIMUTH = 0.0
VIEWER_ELEVATION = -45.0

USE_SECONDARY_IMU = False

INIT_BASE_POS = (5.5, 0.0, 0.8)
INIT_BASE_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)  # yaw=π, 面朝 -x

COMMAND_STEP = (0.2, 0.1, 0.5)
COMMAND_RANGES = {
    "lin_vel_x": (-0.5, 1.5),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-1.0, 1.0),
}
