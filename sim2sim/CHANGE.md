# 变更记录

## 2026-07-10: 构建 libsensor_raycaster.so

**问题：** `sim2sim/` 缺少 `libsensor_raycaster.so`（MuJoCo raycaster 插件），sim2sim 无法启动。

**操作：**
1. 克隆 MuJoCo 3.9.0 源码到 `/tmp/mujoco`
2. 克隆 [Albusgive/mujoco_ray_caster](https://github.com/Albusgive/mujoco_ray_caster) 到 `/tmp/mujoco/plugin/mujoco_ray_caster`
3. 修复两个兼容性问题：
   - 创建 `/tmp/mujoco/include/mujoco/mjtnum.h`（MuJoCo 3.9.0 已将类型定义移至 `mjtype.h`）
   - 修改 `register.cc` 中 `mjPLUGIN_LIB_INIT` 宏添加参数 `sensor_raycaster`
4. 编译并将 `libsensor_raycaster.so` 拷贝到 `sim2sim/`

**依赖安装：** `pip install -e rsl_rl`（本地自定义 rsl_rl 包，sim2sim 推理需要）

---

## 2026-07-10: 强制 vy=0（已撤销，未生效）

**问题：** 机器人启动后 vy 自动漂移到 -0.5，未经指令输入。

**根因：** 训练时 `lin_vel_y` 范围为 `(0.0, 0.0)`，策略从未学习过侧向移动。sim2sim 允许 vy 达到 ±0.5 属于训练分布外行为。

**修改：**

1. `sim2sim_raycaster.py:300` — 观测中强制 vy=0
2. `config.py:53` — 键盘/手柄命令范围限制 vy

**结果：撤销。** 在机器人依旧存在向右偏移的问题，问题可能在高程图 Y 轴翻转方向或地形不对称。

---

## 2026-07-10: 创建 AGENTS.md

在仓库根目录创建 `AGENTS.md`，为 OpenCode 提供项目关键信息：
- 两阶段训练流程（FINETUNE 开关）
- 本地 rsl_rl 安装要求
- ame1.pt / ame2.pt 区别（GlobalEncoder 改变 input_dim）
- sim2sim 对齐要求（raycaster 预热、无效射线、PD 增益）
- Lint 配置（black 120, flake8, isort）
