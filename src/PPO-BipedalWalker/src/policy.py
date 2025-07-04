# walker_ppo/src/models/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

        
class uth_t(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hdim: int = 256, log_std_init: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc_mean = nn.Linear(hdim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, state: torch.Tensor) -> Normal:
        expected_device = next(self.parameters()).device
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.device != expected_device:
            state = state.to(expected_device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        
        return Normal(mean, std)

    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor = None, deterministic: bool = False):
        distribution = self.forward(state)
        
        if deterministic:
            action_pre_tanh = distribution.mean
        elif action is None:
            action_pre_tanh = distribution.sample()
        else:
            action_pre_tanh = action
            
        log_prob = distribution.log_prob(action_pre_tanh).sum(axis=-1)
        scaled_action = torch.tanh(action_pre_tanh)
        
        if action is None and not deterministic:
            entropy = distribution.entropy().sum(axis=-1)
            return scaled_action, log_prob, entropy, action_pre_tanh
        else:
            return scaled_action, log_prob, None, action_pre_tanh