# walker_ppo/src/algo/ppo.py
import torch
import torch.optim as optim
import torch.nn as nn # For MSELoss
import numpy as np

from src.utils import RolloutBuffer, compute_gae_advantages 

class PPOAgent:
    def __init__(self,
                 actor_net: nn.Module,
                 critic_net: nn.Module,
                 obs_dim: int, 
                 action_dim: int, 
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 device: torch.device = torch.device('cuda')):
        
        self.actor = actor_net.to(device)
        self.critic = critic_net.to(device)
        self.device = device

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl 


    def update(self, 
               obs_b: torch.Tensor, 
               actions_b: torch.Tensor,
               log_probs_old_b: torch.Tensor, 
               advantages_b: torch.Tensor, 
               returns_b_target: torch.Tensor):
        
        self.actor.train()
        self.critic.train()

        obs_b = obs_b.to(self.device)
        actions_b = actions_b.to(self.device)
        log_probs_old_b = log_probs_old_b.to(self.device)
        advantages_b = advantages_b.to(self.device)
        returns_b_target = returns_b_target.to(self.device)

        for _ in range(self.ppo_epochs):
            num_samples = obs_b.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                mini_batch_indices = indices[start_idx:end_idx]
                
                obs_mb = obs_b[mini_batch_indices]
                actions_mb = actions_b[mini_batch_indices]
                log_probs_old_mb = log_probs_old_b[mini_batch_indices]
                advantages_mb = advantages_b[mini_batch_indices]
                returns_mb = returns_b_target[mini_batch_indices]

                # Policy Loss
                new_action_distribution = self.actor(obs_mb)
                new_log_probs_mb = new_action_distribution.log_prob(actions_mb).sum(axis=-1)
                entropy_bonus = new_action_distribution.entropy().sum(axis=-1).mean()
                ratio = torch.exp(new_log_probs_mb - log_probs_old_mb)
                
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                current_values_mb = self.critic(obs_mb).squeeze(-1)
                critic_loss = nn.MSELoss()(current_values_mb, returns_mb)
                
                # Total Loss
                loss = policy_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_bonus
                
                # Optimization
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()

        return {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus
        }
