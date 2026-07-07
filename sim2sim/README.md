# AMElocomotion Sim2Sim 部署与地形验证项目

## 一、项目背景

### 1.1 研究目标

在物理仿真环境中实现基于高程图感知的人形机器人稀疏落脚点敏捷运动控制，为后续实机部署建立完整的 Sim2Sim 验证管线。

### 1.2 参考论文

- **AME (Attention-Based Map Encoding)**：基于注意力机制的地形特征编码方案，通过 CNN + Multi-Head Attention 将高维高程图压缩为紧凑特征向量，作为策略网络的地形感知输入。
- **BeamDojo**：基于稀疏落脚点的人形机器人敏捷运动强化学习框架。

### 1.3 技术栈

| 组件 | 技术方案 |
|------|---------|
| 训练框架 | IsaacLab + RSL-RL (PPO) |
| 仿真验证 | MuJoCo (Sim2Sim) |
| 策略网络 | CNN + Multi-Head Attention + MLP (ActorCriticEncoder) |
| 地形感知 | 33×21 网格高程图 (3D 坐标，2079 维) |
| 本体感知 | 关节角度/速度、IMU、速度指令 (96 维) |

---

## 二、项目架构

### 2.1 代码结构

```
AME_Locomotion/
├── pretrained/                 # 预训练策略模型
│   ├── ame1.pt                # CNN+MHA+MLP (无全局特征, input_dim=160)
│   └── ame2.pt                # CNN+MHA+MLP+GlobalEncoder (input_dim=224)
├── rsl_rl/                    # RSL-RL 强化学习框架（含 ActorCriticEncoder）
├── source/ame_locomotion/     # 训练环境配置
│   └── tasks/manager_based/ame_locomotion/
│       ├── 29dof/velocity_env_cfg_29dof.py   # 训练环境配置
│       └── mdp/observations.py               # 观测函数（含 elevation_map）
├── sim2sim/                   # Sim2Sim 验证管线（本项目核心成果）
│   ├── sim2sim_raycaster.py   # 主控制循环
│   ├── mujoco_env.py          # MuJoCo 环境封装
│   ├── policy_inference.py    # 策略推理（AME ActorCriticEncoder）
│   ├── config.py              # 运行配置
│   ├── deploy.yaml            # 部署参数（关节映射/PD增益/观测配置）
│   └── assets/
│       ├── g1_29dof_raycaster.xml       # G1 机器人 MuJoCo 模型
│       ├── plum_blossom_terrain.py      # 梅花桩+独木桥地形生成器
│       └── scene_plum_blossom.xml       # 生成的场景文件
└── scripts/                   # 训练/评估脚本
```

### 2.2 数据流

```
MuJoCo 物理仿真器
│
├── IMU 传感器 (pelvis body)
│   ├── angular velocity → base_ang_vel (3维, scale=0.2)
│   └── quaternion → projected_gravity (3维)
│
├── 关节编码器
│   ├── joint_pos - default → joint_pos_rel (29维)
│   └── joint_vel → joint_vel_rel (29维, scale=0.05)
│
├── Raycaster (torso_link)
│   └── 33×21 射线 → 3D 局部坐标 (693×3 = 2079维)
│
└── 上一步动作 → last_action (29维)
         │
         ▼
    观测向量 (2175维) = proprio(96) + terrain(2079)
         │
         ▼
    ActorCriticEncoder 前向推理
    ├── CNN: (B, 3, 21, 33) → (B, 64, 11, 17)
    ├── Flatten: → (B, 187, 64) local features
    ├── GlobalEncoder: → (B, 64) global feature (ame2 only)
    ├── MHA: Q=proprio(64), K=V=local(187,64) → (B, 64) attention output
    ├── Concat: [global(64)] + mha(64) + proprio(96) → (B, 224)
    └── Actor MLP: 224→512→256→128→29 → raw_action
         │
         ▼
    动作转换: action × 0.25 + default_joint_pos → 关节目标角度
         │
         ▼
    PD 控制 (× decimation 步):
    torque = Kp × (target - pos) - Kd × vel
         │
         ▼
    MuJoCo 物理仿真执行
```

---

## 三、核心工作与技术难点

### 3.1 策略推理重构（AME → AMElocomotion）

**问题**：AME 项目提供的 Sim2Sim 框架仅支持纯 MLP 策略（1415 维输入），无法加载 AMElocomotion 的 CNN + MHA + GlobalEncoder 架构。

**解决方案**：
- 完全重写 `policy_inference.py`，从 checkpoint 的 state_dict 自动解析网络结构
- 支持 ame1.pt（无 GlobalEncoder，input_dim=160）和 ame2.pt（有 GlobalEncoder，input_dim=224）
- 通过 rsl_rl 框架的 `ActorCriticEncoder.act_inference()` 接口进行推理
- 输入格式：字典 `{"policy": tensor}`（而非扁平 tensor）

### 3.2 高程图传感器对齐

**问题**：训练环境（IsaacLab）和验证环境（MuJoCo）的高程图传感器存在多维度差异：

| 差异维度 | IsaacLab 训练 | MuJoCo Sim2Sim |
|----------|--------------|----------------|
| 数据格式 | 3D 坐标 (x,y,z) | 原始仅返回 hit_z |
| 坐标系 | yaw-aligned body frame | 需手动计算 yaw 旋转 |
| 无效射线 | NaN | 返回世界原点 (0,0,0) |
| 网格分辨率 | 0.05m (33×21) | 需手动配置 |
| 空间排列 | 行列方向固定 | 需验证和翻转 |

**解决方案**：
- 新增 `update_3d()` 方法，返回 693×3 的 yaw-aligned 局部坐标
- 无效射线检测（`hit_dist > 0.1`）并设为 `(0, 0, -1.2)`
- z 值 clamp 到 `[-1.2, 0.0]` 对齐训练配置
- Y 轴翻转对齐 IsaacLab 扫描顺序

### 3.3 部署参数对齐

**问题**：deploy.yaml 中多个关键参数与训练配置不匹配：

- action offset：训练时 `use_default_offset=True`，初始值为 0（错误）
- PD 增益：手臂关节 damping 从 10 错配为 1，导致关节振荡
- history_length：训练时为 1（无历史帧），初始误设为 5

**解决方案**：
- 从训练配置 `velocity_env_cfg_29dof.py` 提取所有参数
- 逐项对齐 stiffness/damping/offset/history_length/scale
- 建立参数对照表用于验证

### 3.4 复杂地形构建

**目标场景**：验证策略在稀疏落脚点地形上的运动能力。

**地形设计**：
- 入口平地 → 上行台阶（5层，每层0.2m）→ 平台 → 梅花桩道 → 独木桥 → 平台 → 下行台阶
- 所有结构顶面齐平（z=1.0m）
- 梅花桩：0.25×0.25m，间距0.45m，梅花形交错排列
- 独木桥：0.25m 宽，4m 长

### 3.5 注意力可视化

**目标**：可视化 MHA 的注意力权重，理解策略如何"关注"地形。

**实现**：
- 从 `act_inference()` 提取 attention_weights (187维，11×17)
- 映射到 693 个射线点的 3D 位置
- 彩虹色编码：蓝(低注意力) → 绿 → 黄 → 红(高注意力)
- 实时渲染在 MuJoCo viewer 中

---

## 四、版本迭代记录

基于 git 提交历史，项目经历了以下关键迭代：

### v1 — 首次提交 (`664bdb8`)
- 复制 AME 项目的 Ch2 Baseline Sim2Sim 代码到 AMElocomotion
- 纯 MLP 策略框架（1415 维输入，187 点高程图）
- 尚未适配 AMElocomotion 的 CNN+MHA 策略

### v2 — 策略路径修正 (`c93917e`)
- 修正模型 checkpoint 路径指向
- 为后续策略加载做准备

### v3 — 文档更新 (`1d3519e`)
- 更新 README.md

### v4 — 核心重构 (`b7415b1`)

**这是最重大的一次提交，完成了 Sim2Sim 管线的全面重构：**

- **策略推理重写** (`policy_inference.py`)：
  从纯 MLP 替换为完整的 ActorCriticEncoder 推理管线
  - CNN 地形特征提取
  - Multi-Head Cross-Attention（proprio→terrain）
  - Global Max-Pooling（ame2.pt 特有）
  - 自动从 checkpoint 推断网络结构

- **高程图传感器升级** (`mujoco_env.py`)：
  - 新增 `update_3d()` 方法，输出 693×3 的 3D 局部坐标
  - yaw-aligned 坐标变换
  - 无效射线过滤（检测世界原点返回值）
  - z 轴 clamp [-1.2, 0.0]

- **部署配置重建** (`deploy.yaml`)：
  - 全部 history_length 改为 1（对齐训练配置）
  - action offset 修正为 default_joint_pos
  - PD 增益按训练配置逐项对齐
  - scale 参数对齐（base_ang_vel=0.2, joint_vel=0.05）

- **运行配置修正** (`config.py`)：
  - 模型路径指向 ame1.pt/ame2.pt
  - 网格分辨率 0.05m（33×21 = 693 射线）

### v5 — 关键 Bug 修复 (`f89bd94`)

**修复了导致机器人失控的多个工程 Bug：**

| Bug | 现象 | 修复 |
|-----|------|------|
| offset=0 | 关节扭曲摔倒 | offset 改为 default_joint_pos |
| 手臂 damping=1 | 双手剧烈振荡 | damping 改为 10 |
| 无效射线坐标 | 第一步 action 爆炸 | 无效射线设为 (0, 0, -1.2) |
| Step 0 射线为空 | 高程图全零 | reset 时预热一个控制周期 |
| update_3d 缺少 yaw 旋转 | 坐标系错位 | yaw-aligned 局部坐标变换 |

### v6 — 地形场景 (`44d84b9`)
- 新增梅花桩+独木桥地形生成器
- 台阶→平台→梅花桩→独木桥→平台→台阶 完整路线
- 支持参数化配置（桩尺寸/间距/桥宽等）
- 注意力热力图可视化（彩虹色编码）

---

## 五、当前进度

```
✅ 策略加载与推理（CNN+MHA+MLP）
✅ 高程图 3D 坐标输出（yaw-aligned）
✅ 部署参数完全对齐
✅ 梅花桩+独木桥地形场景
✅ 台阶攀爬验证
✅ 梅花桩行走验证（vx=0.8）
✅ 注意力热力图可视化
🔄 高程图空间排列方向修正（注意力不对称 → 机器人绕圈）
⬜ 纯平地直行验证
⬜ 完整地形全程测试
⬜ Domain Randomization
⬜ 实机部署
```

---

## 六、关键问题与解决记录

### 问题 1：策略输出爆炸（action 范围 ±20+）

**现象**：Sim2Sim 启动后第一步 action 即达到 ±20，机器人瞬间失控。

**根因**：MuJoCo raycaster 未命中的射线返回世界原点 (0,0,0)，在计算局部坐标时 x 值达到 4.0（传感器到世界原点的距离），远超训练时的合法范围，导致 CNN + BatchNorm 输出异常。

**解决**：在 `update_3d()` 中先检测无效射线（`hit_dist > 0.1`），将无效区域设为 `(0, 0, -1.2)`（与训练时一致）。

### 问题 2：action offset 错误

**现象**：机器人启动时关节扭曲、直接摔倒。

**根因**：deploy.yaml 中 `actions.offset = 0.0`，但训练时 `use_default_offset=True`（offset = default_joint_pos）。`target = action × 0.25 + 0.0` 导致目标角度在零点附近而非站立姿态。

**解决**：将 offset 改为 `default_joint_pos` 的值。

### 问题 3：手臂关节振荡

**现象**：双手持续剧烈摆动。

**根因**：deploy.yaml 中手臂关节 damping=1，但训练配置为 damping=10（shoulder/elbow/wrist）。

**解决**：按训练配置修正 damping 值（29个关节逐一核对）。

### 问题 4：Step 0 射线数据为空

**现象**：第一步高程图全为零，策略输出错误动作。

**根因**：MuJoCo raycaster 插件在首次 `mj_step` 之前不计算射线。

**解决**：在 `reset()` 中先执行一个完整控制周期进行射线预热。

### 问题 5：行走时 yaw 漂移（绕圈）

**现象**：给前进指令后机器人缓慢顺时针旋转，走弧线。

**根因分析**（多轮诊断）：
1. 纯 IMU/gyro 数据正常（无持续角速度偏差）
2. Camera 和 IMU 坐标系对齐正确（forward 差 < 2°）
3. **注意力热力图左右不对称**（偏右下角）→ 高程图空间排列方向与训练时不一致

**当前状态**：正在排查 Y 轴翻转方向（4种组合逐个测试）。

---

## 七、技术指标对照表

| 指标 | 训练环境 | Sim2Sim 当前 |
|------|---------|-------------|
| 物理引擎 | PhysX | MuJoCo |
| 仿真步长 | 0.005s | 0.002s |
| Decimation | 4 | 10 |
| 策略频率 | 50Hz | 50Hz |
| 观测维度 | 2175 | 2175 |
| 高程图分辨率 | 0.05m (33×21) | 0.05m (33×21) |
| 高程图数据 | 3D 坐标 (x,y,z) | 3D 坐标 (x,y,z) |
| 策略网络 | CNN+MHA+MLP | CNN+MHA+MLP |
| action scale | 0.25 | 0.25 |
| action offset | default_joint_pos | default_joint_pos |
| history_length | 1 | 1 |

---

## 八、后续计划

### 短期（1-2 周）
1. 修正高程图空间排列方向，解决 yaw 漂移
2. 完成梅花桩+独木桥全程测试
3. 建立性能评估指标（成功率、平均速度、偏移量）

### 中期（2-4 周）
1. 引入 Domain Randomization（高程图噪声、时延模拟、空间位移）
2. 测试不同策略模型（ame1 vs ame2）的性能差异
3. 优化推理延迟（ONNX/TensorRT 导出）

### 长期（1-2 月）
1. 实机部署（Jetson + Unitree G1）
2. 点云→高程图的在线生成管线
3. 稀疏落脚点场景的闭环验证
