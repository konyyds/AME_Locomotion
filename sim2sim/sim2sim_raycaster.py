import argparse
import os
import time
from collections import deque

import numpy as np
import yaml

try:
    import glfw
except Exception:  # pragma: no cover - glfw is unavailable in --no-viewer mode.
    glfw = None

import config
from mujoco_env import MujocoRaycasterEnv
from policy_inference import PolicyInference as Inference


OBSERVATION_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "height_scanner",
)


def key(name: str, fallback: int) -> int:
    """Use GLFW key codes when viewer is available, otherwise use stable values."""
    if glfw is None:
        return fallback
    return getattr(glfw, f"KEY_{name}", fallback)


def quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation. MuJoCo IMU quaternion uses wxyz order."""
    shape = vec.shape
    quat = quat_wxyz.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2.0
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


class RaycasterSim2Sim:
    def __init__(self, args):
        self.args = args
        self.command = np.array([-0.5, 0.0, 0.0], dtype=np.float32)

        # MuJoCo scene
        self.env = MujocoRaycasterEnv(
            scene_path=config.ROBOT_SCENE,
            plugin_library=config.RAYCASTER_PLUGIN_LIBRARY,
            sim_dt=config.SIM_DT,
        )
        
        with open(config.DEPLOY_CONFIG, "r", encoding="utf-8") as f:
            self.deploy_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

        self.policy = Inference(config.POLICY_PATH, device=config.DEVICE)

        # Policy runs once every DECIMATION MuJoCo simulation steps.
        self.decimation = config.DECIMATION
        self.policy_dt = config.SIM_DT * self.decimation
        deploy_policy_dt = float(self.deploy_cfg["step_dt"])
        if abs(self.policy_dt - deploy_policy_dt) > 1e-6:
            raise ValueError(
                f"config policy_dt {self.policy_dt} != deploy step_dt {deploy_policy_dt}"
            )

        self.num_joints = self.env.num_joints

        # deploy.yaml stores policy/asset joint order. MuJoCo sensors use SDK order.
        self.asset_to_sdk = np.asarray(
            self.deploy_cfg["joint_ids_map"], dtype=np.int64
        )
        self.default_joint_pos_asset = np.asarray(
            self.deploy_cfg["default_joint_pos"], dtype=np.float32
        )
        self.default_joint_pos_sdk = np.zeros(self.num_joints, dtype=np.float32)
        self.default_joint_pos_sdk[self.asset_to_sdk] = self.default_joint_pos_asset

        action_cfg = self.deploy_cfg["actions"]["JointPositionAction"]
        raw_scale = action_cfg["scale"]
        raw_offset = action_cfg["offset"]
        self.action_scale_asset = np.full(
            self.num_joints, float(raw_scale), dtype=np.float32
        ) if isinstance(raw_scale, (int, float)) else np.asarray(raw_scale, dtype=np.float32)
        self.action_offset_asset = np.full(
            self.num_joints, float(raw_offset), dtype=np.float32
        ) if isinstance(raw_offset, (int, float)) else np.asarray(raw_offset, dtype=np.float32)

        self.stiffness = np.asarray(self.deploy_cfg["stiffness"], dtype=np.float32)
        self.damping= np.asarray(self.deploy_cfg["damping"], dtype=np.float32)
        self.torque_min = self.env.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.torque_max = self.env.model.actuator_ctrlrange[:, 1].astype(np.float32)

        self.last_action = np.zeros(self.num_joints, dtype=np.float32)

        imu_prefix = "secondary_imu" if config.USE_SECONDARY_IMU else "imu"
        self.imu_quat_adr = self.env.sensor_address(f"{imu_prefix}_quat")
        self.imu_gyro_adr = self.env.sensor_address(f"{imu_prefix}_gyro")

        # Observation order, scale, clip, and history length follow deploy.yaml.
        self.observation_terms = []
        for name in self.deploy_cfg["observations"].keys():
            if name in OBSERVATION_ORDER:
                self.observation_terms.append(name)

        # Each observation term has its own history length in deploy.yaml.
        self.history = {}
        for name in self.observation_terms:
            self.history[name] = deque(maxlen=self._history_length(name))

        self.reset()
        self.print_startup_info(config.ROBOT_SCENE)
        self.print_keyboard_help()
        if not args.no_viewer:
            self.env.launch_viewer(key_callback=self.key_callback)

    def _history_length(self, term_name: str) -> int:
        return int(self.deploy_cfg["observations"][term_name].get("history_length", 1))

    def _scale_and_clip(self, term_name: str, obs: np.ndarray) -> np.ndarray:
        term_cfg = self.deploy_cfg["observations"][term_name]
        obs = obs.astype(np.float32, copy=True)
        clip = term_cfg.get("clip")
        if clip is not None:
            obs = np.clip(obs, clip[0], clip[1])
        scale = term_cfg.get("scale")
        if scale is not None:
            obs *= np.asarray(scale, dtype=np.float32)
        return obs

    def reset(self):
        self.command[:] = 0.0
        self.last_action[:] = 0.0
        self.env.reset(self.default_joint_pos_sdk)
        self.reset_observation_history()
        print("Reset simulation state.")

    def reset_observation_history(self):
        for history in self.history.values():
            history.clear()

        initial_terms = self.update_observation()
        for name, value in initial_terms.items():
            for _ in range(self._history_length(name) - 1):
                self.history[name].append(value.copy())
    
    def key_callback(self, keycode: int):
        vx_step, _, yaw_step = config.COMMAND_STEP
        handled = True
        if keycode in (key("UP", 265), key("KP_8", 328)):
            self.command[0] += vx_step
        elif keycode in (key("DOWN", 264), key("KP_2", 322)):
            self.command[0] -= vx_step
        elif keycode in (key("LEFT", 263), key("KP_4", 324)):
            self.command[2] += yaw_step
        elif keycode in (key("RIGHT", 262), key("KP_6", 326)):
            self.command[2] -= yaw_step
        elif keycode in (key("SPACE", 32), key("KP_5", 325)):
            self.command[:] = 0.0
        elif keycode == key("R", 82):
            self.reset()
        else:
            handled = False

        if handled:
            ranges = config.COMMAND_RANGES
            self.command[0] = np.clip(self.command[0], *ranges["lin_vel_x"])
            self.command[1] = np.clip(self.command[1], *ranges["lin_vel_y"])
            self.command[2] = np.clip(self.command[2], *ranges["ang_vel_z"])
            print(
                "Command "
                f"vx={self.command[0]: .2f}, "
                f"vy={self.command[1]: .2f}, "
                f"yaw={self.command[2]: .2f}"
            )

    def print_keyboard_help(self):
        print(
            "Keyboard: Up/Down or KP8/KP2 vx, Left/Right or KP4/KP6 yaw, "
            "Space/KP5 stop, R reset."
        )

    def print_startup_info(self, scene_path: str):
        obs_dim = 0
        for name in self.observation_terms:
            history_length = len(self.history[name])
            single_obs_dim = len(self.history[name][0])
            obs_dim += history_length * single_obs_dim
            
        print("Loaded rough raycaster sim2sim.")
        print(f"  policy: {self.policy}")
        print(f"  scene: {scene_path}")
        print(
            f"  policy_dt={self.policy_dt:.3f}s "
            f"sim_dt={config.SIM_DT:.3f}s "
            f"decimation={self.decimation}"
        )
        print(
            "  height_scanner: "
            f"h={self.env.height_scanner.h_ray_num} "
            f"v={self.env.height_scanner.v_ray_num} "
            f"dim3d={self.env.height_scanner.height_scan_3d.size}"
        )
        print(f"  obs_dim={obs_dim}")
        print(f"\n=== CRITICAL: Action Offset Check ===")
        print(f"  deploy.yaml offset value: {self.action_offset_asset[:5]}")
        print(f"  Training config: use_default_offset=True")
        print(f"  => Training uses default_joint_pos as offset!")
        print(f"  => deploy.yaml offset=0.0 is WRONG if training used default_joint_pos")

    def update_observation(self) -> dict[str, np.ndarray]:
        sensordata = self.env.data.sensordata
        joint_pos_sdk = sensordata[: self.num_joints].astype(np.float32)
        joint_vel_sdk = sensordata[
            self.num_joints : 2 * self.num_joints
        ].astype(np.float32)

        imu_gyro = sensordata[self.imu_gyro_adr : self.imu_gyro_adr + 3].astype(np.float32)
        imu_quat = sensordata[self.imu_quat_adr : self.imu_quat_adr + 4].astype(np.float32)
        projected_gravity = quat_apply_inverse(
            imu_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        velocity_commands = self.command.copy()
        # 关节顺序从 sdk 顺序重新排列成 policy 训练时的 asset 顺序
        joint_pos_asset = joint_pos_sdk[self.asset_to_sdk]
        joint_vel_asset = joint_vel_sdk[self.asset_to_sdk]
        last_action = self.last_action.copy()
        height_scanner = self.env.height_scanner.update_3d().copy()

        raw_obs_terms = {
            "base_ang_vel": imu_gyro,
            "projected_gravity": projected_gravity,
            "velocity_commands": velocity_commands,
            "joint_pos_rel": joint_pos_asset - self.default_joint_pos_asset,
            "joint_vel_rel": joint_vel_asset,
            "last_action": last_action,
            "height_scanner": height_scanner,
        }

        cur_obs_terms = {}
        for name in self.observation_terms:
            cur_obs_terms[name] = self._scale_and_clip(name, raw_obs_terms[name])

        for name, value in cur_obs_terms.items():
            self.history[name].append(value)
        return cur_obs_terms

    def get_history_obs(self) -> np.ndarray:
        # 把每个观测项的历史帧按顺序拼成一个一维数组，作为 policy 的最终输入 obs
        history_obs = []
        for name in self.observation_terms:
            for obs_frame in self.history[name]:
                history_obs.append(obs_frame)
        return np.concatenate(history_obs, dtype=np.float32)

    def action_to_joint_pos_sdk(self, raw_action_asset: np.ndarray, step_count: int = 0) -> np.ndarray:
        self.last_action[:] = raw_action_asset
        action_asset = raw_action_asset * self.action_scale_asset + self.action_offset_asset
        action_sdk = np.zeros(self.num_joints, dtype=np.float32)
        action_sdk[self.asset_to_sdk] = action_asset
        if step_count % 100 == 0:
            print(f"[action] raw={raw_action_asset[:5]}")
            print(f"[action] scale={self.action_scale_asset[:5]}")
            print(f"[action] offset={self.action_offset_asset[:5]}")
            print(f"[action] target_sdk={action_sdk[:5]}")
        return action_sdk

    def compute_torque(self, target_joint_pos_sdk: np.ndarray) -> np.ndarray:
        joint_pos_sdk = self.env.data.sensordata[: self.num_joints].astype(np.float32)
        joint_vel_sdk = self.env.data.sensordata[
            self.num_joints : 2 * self.num_joints
        ].astype(np.float32)
        # PD control
        torque = (
            self.stiffness * (target_joint_pos_sdk - joint_pos_sdk)
            - self.damping* joint_vel_sdk
        )
        return np.clip(torque, self.torque_min, self.torque_max)

    def run(self, steps: int | None = None):
        step_count = 0
        while self.env.is_running():
            start_time = time.perf_counter()

            raw_obs = self.update_observation()
            obs = self.get_history_obs()
            # === DEBUG START ===
            if step_count < 5 or step_count % 100 == 0:
                height = obs[-2079:]
                proprio = obs[:-2079]
                print(f"[DEBUG step={step_count}]")
                print(f"  proprio({len(proprio)}): {proprio[:6]}... (first 6)")
                print(f"  height({len(height)}): min={height.min():.3f} max={height.max():.3f} "
                      f"mean={height.mean():.3f} nan_count={np.isnan(height).sum()}")
                print(f"  command: vx={self.command[0]:.2f} yaw={self.command[2]:.2f}")
                print(f"  joint_pos_rel range: [{raw_obs.get('joint_pos_rel', np.array([0])).min():.3f}, "
                      f"{raw_obs.get('joint_pos_rel', np.array([0])).max():.3f}]")
            # === DEBUG END ===
            raw_action = self.policy(obs)
            if raw_action.shape[0] != self.num_joints:
                raise ValueError(
                    f"Policy action dim {raw_action.shape[0]} != {self.num_joints}"
                )
            # === DEBUG START ===
            if step_count < 5 or step_count % 100 == 0:
                print(f"  action: min={raw_action.min():.3f} max={raw_action.max():.3f} mean={raw_action.mean():.3f}")
            # === DEBUG END ===

            target_joint_pos_sdk = self.action_to_joint_pos_sdk(raw_action, step_count)
            for _ in range(self.decimation):
                torque = self.compute_torque(target_joint_pos_sdk)
                self.env.data.ctrl[:] = torque
                self.env.step()
            if step_count % 50 == 0:
                joint_pos_sdk = self.env.data.sensordata[: self.num_joints].astype(np.float32)
                print(f"[PD] target_sdk前5={target_joint_pos_sdk[:5]}")
                print(f"[PD] actual_sdk前5={joint_pos_sdk[:5]}")
                print(f"[PD] torque前5={torque[:5]}")
            if step_count < 5:
                joint_pos_sdk = self.env.data.sensordata[: self.num_joints].astype(np.float32)
                print(f"[step={step_count}] action前5={raw_action[:5]}")
                print(f"[step={step_count}] torque前5={torque[:5]}")
                print(f"[step={step_count}] joint_pos_sdk前5={joint_pos_sdk[:5]}")
                print(f"[step={step_count}] target前5={target_joint_pos_sdk[:5]}")

            step_count += 1

            self.env.draw_debug()
            self.env.sync()

            if steps is not None and step_count >= steps:
                break

            sleep_time = self.policy_dt - (time.perf_counter() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    
    required_paths = (
        config.POLICY_PATH,
        config.DEPLOY_CONFIG,
        config.ROBOT_SCENE,
        config.RAYCASTER_PLUGIN_LIBRARY,
    )
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    app = RaycasterSim2Sim(args)
    app.run(steps=args.steps)


if __name__ == "__main__":
    main()
