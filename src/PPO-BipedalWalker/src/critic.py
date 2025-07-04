# project_root/src/models/critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hdim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc_value = nn.Linear(hdim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        expected_device = next(self.parameters()).device
        
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.device != expected_device:
            state = state.to(expected_device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        return value