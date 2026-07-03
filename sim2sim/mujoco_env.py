import mujoco
import mujoco.viewer
import numpy as np

import config


class HeightScanner:
    def __init__(self, model, data, sensor_name: str, camera_name: str):
        self.model = model
        self.data = data
        self.sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if self.sensor_id < 0:
            raise ValueError(f"Sensor '{sensor_name}' not found")
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.camera_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found")

        self.sensor_adr = int(model.sensor_adr[self.sensor_id])
        self.sensor_dim = int(model.sensor_dim[self.sensor_id])
        self.pos_w_start = 0
        self.pos_w_size = self.sensor_dim
        self.h_ray_num = 0
        self.v_ray_num = 0
        self._refresh_plugin_info()
        if self.h_ray_num <= 0 or self.v_ray_num <= 0:
            self.h_ray_num = int(round(self.pos_w_size / 3))
            self.v_ray_num = 1

        expected_size = self.h_ray_num * self.v_ray_num * 3
        if self.pos_w_size != expected_size:
            raise ValueError(
                "Height scanner pos_w size mismatch: "
                f"got {self.pos_w_size}, expected {expected_size}"
            )

        self.height_scan = np.zeros(self.h_ray_num * self.v_ray_num, dtype=np.float32)
        self.height_scan_3d = np.zeros(self.h_ray_num * self.v_ray_num * 3, dtype=np.float32)
        self.hit_positions = np.full(
            (self.h_ray_num * self.v_ray_num, 3), np.nan, dtype=np.float64
        )
        self._step_counter = 0

    def _refresh_plugin_info(self):
        sensor_plugin_id = int(self.model.sensor_plugin[self.sensor_id])
        if sensor_plugin_id < 0:
            return
        state_idx = int(self.model.plugin_stateadr[sensor_plugin_id])
        state_num = int(self.model.plugin_statenum[sensor_plugin_id])
        if state_idx < 0 or state_num < 4:
            return
        h_ray_num = int(self.data.plugin_state[state_idx])
        v_ray_num = int(self.data.plugin_state[state_idx + 1])
        if h_ray_num > 0 and v_ray_num > 0:
            self.h_ray_num = h_ray_num
            self.v_ray_num = v_ray_num
        self.pos_w_start = int(self.data.plugin_state[state_idx + 2])
        self.pos_w_size = int(self.data.plugin_state[state_idx + 3])

    def update(self) -> np.ndarray:
        start = self.sensor_adr + self.pos_w_start
        end = start + self.pos_w_size
        pos_w = self.data.sensordata[start:end].reshape(-1, 3)
        self.hit_positions[:] = pos_w

        sensor_z = self.data.cam_xpos[self.camera_id, 2]
        height_scan = sensor_z - pos_w[:, 2] - config.HEIGHT_SCANNER_OFFSET
        height_scan = np.nan_to_num(
            height_scan,
            nan=config.HEIGHT_SCANNER_CLIP[1],
            posinf=config.HEIGHT_SCANNER_CLIP[1],
            neginf=config.HEIGHT_SCANNER_CLIP[0],
        )
        height_scan = np.clip(
            height_scan,
            config.HEIGHT_SCANNER_CLIP[0],
            config.HEIGHT_SCANNER_CLIP[1],
        ).astype(np.float32)

        if config.HEIGHT_SCANNER_FLIP_Y_FOR_ISAACLAB_ORDER:
            height_scan = height_scan.reshape(self.v_ray_num, self.h_ray_num)[::-1]
            height_scan = height_scan.reshape(-1)

        self.height_scan[:] = height_scan

        # flat_height = sensor_z - config.HEIGHT_SCANNER_OFFSET
        # self.height_scan[:] = flat_height

        return self.height_scan

    def update_3d(self) -> np.ndarray:
        start = self.sensor_adr + self.pos_w_start
        end = start + self.pos_w_size
        pos_w = self.data.sensordata[start:end].reshape(-1, 3)

        # 检测无效射线（MuJoCo 未命中的射线返回世界原点）
        hit_dist = np.linalg.norm(pos_w, axis=1)
        valid_mask = hit_dist > 0.1

        # 传感器位置：torso 本体，不加 20m 偏移
        # IsaacLab 的 sensor.data.pos_w 就是 torso 位置，不含 offset
        sensor_pos = self.data.cam_xpos[self.camera_id]
        rot_mat = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)

        rel = pos_w - sensor_pos
        local_x =  c * rel[:, 0] + s * rel[:, 1]
        local_y = -s * rel[:, 0] + c * rel[:, 1]
        local_z = rel[:, 2]

        local = np.stack([local_x, local_y, local_z], axis=-1)

        # 保存射线命中位置（用于可视化）
        self.hit_positions[:] = pos_w

        # 无效射线设为 (0, 0, -1.2)
        local[~valid_mask] = 0.0
        local[~valid_mask, 2] = -1.2

        # z 裁剪
        local[:, 2] = np.clip(local[:, 2], -1.2, 0.0)

        # Y 轴翻转
        if config.HEIGHT_SCANNER_FLIP_Y_FOR_ISAACLAB_ORDER:
            local = local.reshape(self.v_ray_num, self.h_ray_num, 3)[::-1]

        self._step_counter += 1
        return local.reshape(-1).astype(np.float32)

    def draw_points(self, user_scn):
        if user_scn is None:
            return
        size = np.array([config.HEIGHT_SCANNER_POINT_RADIUS] * 3, dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        rgba = np.array(config.HEIGHT_SCANNER_POINT_RGBA, dtype=np.float32)

        for pos in self.hit_positions:
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            if not np.all(np.isfinite(pos)):
                continue
            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba
            )
            user_scn.ngeom += 1


class MujocoRaycasterEnv:
    def __init__(
        self,
        scene_path: str,
        plugin_library: str,
        sim_dt: float,
    ):
        mujoco.mj_loadPluginLibrary(plugin_library)
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.num_joints = self.model.nu
        self.viewer = None

        self.height_scanner = HeightScanner(
            self.model,
            self.data,
            config.HEIGHT_SCANNER_SENSOR_NAME,
            config.HEIGHT_SCANNER_CAMERA_NAME,
        )

    def sensor_address(self, sensor_name: str) -> int:
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise ValueError(f"Sensor '{sensor_name}' not found")
        return int(self.model.sensor_adr[sensor_id])

    def reset(self, default_joint_pos_sdk: np.ndarray):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:3] = np.asarray(config.INIT_BASE_POS, dtype=np.float64)
        self.data.qpos[3:7] = np.asarray(config.INIT_BASE_QUAT_WXYZ, dtype=np.float64)
        self.data.qpos[7 : 7 + self.num_joints] = default_joint_pos_sdk
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.height_scanner.update()

    def launch_viewer(self, key_callback=None):
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_callback
        )
        self.viewer.cam.lookat[:] = np.asarray(config.VIEWER_LOOKAT, dtype=np.float64)
        self.viewer.cam.distance = config.VIEWER_DISTANCE
        self.viewer.cam.azimuth = config.VIEWER_AZIMUTH
        self.viewer.cam.elevation = config.VIEWER_ELEVATION

    def is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def sync(self):
        if self.viewer is None:
            return
        self.viewer.sync()

    def draw_debug(self):
        if self.viewer is None:
            return
        self.viewer.user_scn.ngeom = 0
        if config.DRAW_HEIGHT_SCANNER_POINTS:
            self.height_scanner.draw_points(self.viewer.user_scn)
