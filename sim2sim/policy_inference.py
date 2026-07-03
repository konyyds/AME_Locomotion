import sys
sys.path.insert(0, "/home/frank/Desktop/AME_Locomotion/rsl_rl")

import torch
import numpy as np
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder


class PolicyInference:
    def __init__(self, policy_file, device="cpu"):
        self.policy_file = policy_file
        self.device = torch.device(device)
        ckpt = torch.load(policy_file, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        num_actions = int(state_dict["std"].shape[0])
        proprio_dim = int(state_dict["actor_proprio_embedding.weight"].shape[1])
        terrain_dim = 33 * 21 * 3

        critic_proprio_dim = proprio_dim + 3  # +base_lin_vel
        obs_dummy = {
            "policy": torch.zeros(1, proprio_dim + terrain_dim),
            "critic": torch.zeros(1, critic_proprio_dim + terrain_dim),
        }
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        self.policy = ActorCriticEncoder(
            obs=obs_dummy,
            obs_groups=obs_groups,
            num_actions=num_actions,
            map_scan_dim=(33, 21, 3),
            cnn_downsample=True,
            attach_global="global_encoder.0.weight" in state_dict,
        ).to(self.device)

        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()

    def forward(self, obs):
        t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.act_inference({"policy": t})
        return action.cpu().numpy().reshape(-1).astype(np.float32)

    def __call__(self, obs):
        return self.forward(obs)

    def __str__(self):
        return f"AME CNN+MHA ({self.policy_file})"
