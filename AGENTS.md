# AGENTS.md

## What this is

Isaac Lab + RSL-RL project reproducing AME (Attention-Based Map Encoding) for Unitree G1 29-DoF humanoid locomotion. Two main components: training (Isaac Sim) and sim2sim validation (MuJoCo).

## Critical setup

Two editable installs are required and order matters:

```bash
python -m pip install -e source/ame_locomotion   # Isaac Lab extension
python -m pip install -e rsl_rl                   # CUSTOM rsl_rl, not the upstream one
```

The local `rsl_rl` overrides the upstream package with a custom `ActorCriticEncoder` network. Without it, imports fail silently or crash at runtime.

## Two-stage training

1. Run `bash run_train.sh` (stage 1, `FINETUNE = False`).
2. After stage 1 completes, set `FINETUNE = True` in `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py:30`.
3. Run `bash run_train.sh` again (stage 2).

Stage 2 enables domain randomization (push, mass, COM), observation noise, and different reward weights. Setting this wrong produces a silently broken training run.

## Commands

- Train: `bash run_train.sh`
- Play/eval: `bash run_play.sh` (uses `pretrained/ame1.pt` by default)
- Lint: `pre-commit run --all-files` (black 120 chars, flake8, isort)
- No test suite exists. Validation is done via play scripts and sim2sim.

## Key files

| File | Role |
|------|------|
| `rsl_rl/rsl_rl/modules/actor_critic_encoder.py` | AME network: CNN + MHA + optional GlobalEncoder |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py` | Training env config (scene, MDP, rewards, FINETUNE flag) |
| `scripts/rsl_rl/train.py` | Training entrypoint |
| `scripts/rsl_rl/play.py` | Evaluation entrypoint |
| `sim2sim/` | MuJoCo sim2sim validation pipeline |
| `pretrained/ame1.pt` | Checkpoint without global context (input_dim=160) |
| `pretrained/ame2.pt` | Checkpoint with global context (input_dim=224) |

## Architecture gotchas

- CNN input is full xyz coordinates (not z-only height), different from the original paper.
- CNN uses stride-based downsampling to shorten MHA sequence length.
- `ame1.pt` vs `ame2.pt`: the presence of `GlobalEncoder` changes the actor input dimension (160 vs 224). The sim2sim `policy_inference.py` auto-detects this from the checkpoint state_dict.
- Elevation map is 33x21 grid at 0.05m resolution (693 rays, 2079-dim when using xyz).
- Action space: `action * 0.25 + default_joint_pos`. The `use_default_offset=True` in the action config is essential; setting offset to 0 breaks the robot.

## Sim2sim notes

The sim2sim pipeline (`sim2sim/`) has several non-obvious alignment requirements documented in `sim2sim/README.md`:
- MuJoCo raycaster must be warmed up with one `mj_step` before reading rays (step 0 returns empty).
- Invalid rays must be set to `(0, 0, -1.2)` not `(0, 0, 0)`.
- Y-axis may need flipping to match IsaacLab scan order (currently under investigation).
- PD gains and action offset must match the training config exactly.

## Linting

Pre-commit enforces: black (line-length=120), flake8 (with simplify/return plugins), isort (black profile), pyupgrade (py310+), license headers. Pyright is commented out in the pre-commit config.
