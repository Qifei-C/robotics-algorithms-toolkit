# walker_ppo/src/utils.py
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil

def set_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device

def set_seed(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    if env is not None and hasattr(env, 'seed'):
        env.seed(seed)
        
    print(f"Set seed to {seed}")

def setup_logger(log_dir: str = "logs/", 
                 experiment_name: str = "experiment", 
                 config_dict: dict = None,
                 use_tensorboard: bool = True) -> tuple[SummaryWriter | None, str]:
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    experiment_path = os.path.join(log_dir, experiment_name)
    idx = 0
    unique_experiment_path = experiment_path
    while os.path.exists(unique_experiment_path): 
        idx += 1
        unique_experiment_path = f"{experiment_path}_{idx}"
    
    os.makedirs(unique_experiment_path, exist_ok=True)
    print(f"Logging to: {unique_experiment_path}")

    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=unique_experiment_path)
        if config_dict is not None and writer is not None:
            hparams_str = "\n".join([f"{key}: {value}" for key, value in config_dict.items()])
            writer.add_text("hyperparameters", hparams_str)

    return writer, unique_experiment_path

def save_checkpoint(state: dict, 
                    checkpoint_dir: str = "checkpoints/", 
                    filename: str = "checkpoint.pth.tar", 
                    is_best: bool = False,
                    experiment_name: str = "experiment"):
    
    exp_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    if not os.path.exists(exp_checkpoint_dir):
        os.makedirs(exp_checkpoint_dir)
    
    filepath = os.path.join(exp_checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    if is_best:
        best_filepath = os.path.join(exp_checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model checkpoint saved to {best_filepath}")

def load_checkpoint(checkpoint_path: str, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer = None, 
                    device: str = 'cpu') -> dict:
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


class RolloutBuffer:
    def __init__(self, num_steps: int, obs_dim: tuple, action_dim: tuple, device: torch.device):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize buffers
        self.observations = torch.zeros((self.num_steps, *self.obs_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.num_steps, *self.action_dim), dtype=torch.float32).to(self.device)
        self.log_probs = torch.zeros((self.num_steps,), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.num_steps,), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((self.num_steps,), dtype=torch.float32).to(self.device) 
        self.values = torch.zeros((self.num_steps,), dtype=torch.float32).to(self.device) 
        
        self.ptr = 0 #
        self.path_start_idx = 0 

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, value: float, log_prob: float):
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32).to(self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32).to(self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32).to(self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32).to(self.device)
        
        self.ptr += 1

    def get_all_data(self) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[:self.ptr],
            "actions": self.actions[:self.ptr],
            "log_probs": self.log_probs[:self.ptr],
            "rewards": self.rewards[:self.ptr],
            "dones": self.dones[:self.ptr],
            "values": self.values[:self.ptr],
        }
    
    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0

    def is_full(self) -> bool:
        return self.ptr >= self.num_steps


def compute_gae_advantages(rewards: torch.Tensor, 
                           values: torch.Tensor, 
                           dones: torch.Tensor, 
                           gamma: float, 
                           gae_lambda: float, 
                           bootstrap_value: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    
    if values.shape[0] == num_steps:
        if bootstrap_value is None:
            raise ValueError("bootstrap_value must be provided if len(values) == len(rewards)")
        next_values = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    elif values.shape[0] == num_steps + 1:
        next_values = values[1:]
        values = values[:-1]
    else:
        raise ValueError(f"Shape mismatch for values. Expected ({num_steps},) or ({num_steps+1},), got {values.shape}")

    last_gae_lam = 0
    for t in reversed(range(num_steps)):
        next_non_terminal = 1.0 - dones[t] 
        delta = rewards[t] + gamma * next_values[t] * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    
    returns = advantages + values
    return advantages, returns


class RunningMeanStd:
    def __init__(self, shape: tuple | int = (), epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = epsilon

        self.epsilon = epsilon
        self._shape = shape if isinstance(shape, tuple) else (shape,)

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0, dtype=torch.float64)
        batch_var = torch.var(x, dim=0, unbiased=False, dtype=torch.float64)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count
        self.mean = new_mean
        self.count = tot_count

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var)

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def normalize(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(x)
        
        current_mean = self.mean.to(x.device, dtype=x.dtype)
        current_std = self.std.to(x.device, dtype=x.dtype)
        
        return (x - current_mean) / (current_std + self.epsilon)

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        current_mean = self.mean.to(x_norm.device, dtype=x_norm.dtype)
        current_std = self.std.to(x_norm.device, dtype=x_norm.dtype)
        return x_norm * (current_std + self.epsilon) + current_mean