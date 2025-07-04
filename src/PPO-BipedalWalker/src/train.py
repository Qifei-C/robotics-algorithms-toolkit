# walker_ppo/train.py
import numpy as np
import torch
import yaml
import argparse
import os
from tqdm import tqdm

from src.models.policy import uth_t
from src.models.critic import Critic
from src.envs.walker_env import create_walker_env, preprocess_observation
from src.algo.ppo import PPOAgent
from src.utils import set_device, set_seed, setup_logger, save_checkpoint, RolloutBuffer, compute_gae_advantages

def run_evaluation_episode(env, policy_network, preprocess_obs_fn, device, max_steps=1000):
    time_step = env.reset()
    current_processed_obs = preprocess_obs_fn(time_step.observation)
    
    episode_reward = 0
    done = False
    truncated = False
    
    policy_network.eval()
    
    for _ in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(current_processed_obs, dtype=torch.float32).unsqueeze(0).to(device)
            scaled_action, _, _, _ = policy_network.get_action_log_prob(obs_tensor, deterministic=True)
        
        action_to_env = scaled_action.cpu().numpy().flatten()
        
        time_step = env.step(action_to_env)
        
        next_processed_obs = preprocess_obs_fn(time_step.observation)
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        
        episode_reward += reward
        current_processed_obs = next_processed_obs
        
        if done:
            break
            
    return episode_reward


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None: config = {}

    # Setup
    device = set_device(config.get("device", "auto"))
    env_seed = config.get('env_seed', 42)
    set_seed(env_seed)
    
    # Logger
    experiment_name = config.get("experiment_name", "ppo_walker_experiment")
    writer, log_path = setup_logger(
        log_dir=config.get("log_dir", "logs/"), 
        experiment_name=experiment_name,
        config_dict=config
    )
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/")
    
    task_name = config.get('task_name', 'walk')
    reward_wrapper_config = config.get('reward_wrapper', {})

    env, obs_dim, action_dim = create_walker_env(
        seed=env_seed, 
        task_name=task_name,
        reward_wrapper_config=reward_wrapper_config
    )
    print(f"Environment: Walker '{task_name}' | Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    if reward_wrapper_config.get("use_custom_wrapper", False):
        print("Using custom reward wrapper with specified parameters.")
    else:
        print("Using default environment rewards.")

    # Networks
    actor_hdim = config.get("actor_hdim", 256)
    critic_hdim = config.get("critic_hdim", 256)
    log_std_init_val = config.get("log_std_init", -0.5)
    
    actor = uth_t(obs_dim, action_dim, hdim=actor_hdim, log_std_init=log_std_init_val).to(device)
    critic = Critic(obs_dim, hdim=critic_hdim).to(device)
    print("uth_t and Critic networks initialized.")

    # PPO Agent
    ppo_config = config.get("ppo_agent", {})
    rollout_steps_N = ppo_config.get("rollout_steps", 4096)
    agent = PPOAgent(
        actor_net=actor,
        critic_net=critic,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=ppo_config.get("lr_actor", 1e-4),
        lr_critic=ppo_config.get("lr_critic", 3e-4),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_epsilon=ppo_config.get("clip_epsilon", 0.1),
        ppo_epochs=ppo_config.get("ppo_epochs", 10),
        mini_batch_size=ppo_config.get("mini_batch_size", 128),
        value_loss_coef=ppo_config.get("value_loss_coef", 0.5),
        entropy_coef=ppo_config.get("entropy_coef", 0.005),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        device=device
    )

    rollout_steps_N = ppo_config.get("rollout_steps", 4096)
    _obs_dim_tuple = (obs_dim,) if isinstance(obs_dim, int) else tuple(obs_dim)
    _action_dim_tuple = (action_dim,) if isinstance(action_dim, int) else tuple(action_dim)

    rollout_buffer = RolloutBuffer(
        num_steps=rollout_steps_N,
        obs_dim=_obs_dim_tuple,
        action_dim=_action_dim_tuple,
        device=device
    )


    # Training Loop
    total_timesteps = config.get("total_timesteps", 1_000_000)
    eval_interval_steps = config.get("eval_interval_steps", 20000) 
    log_interval_updates = config.get("log_interval_updates", 1)
    save_interval_updates = config.get("save_interval_updates", 50)
    
    max_episode_len = config.get("max_episode_len", 1000)

    global_step_count = 0
    ppo_update_count = 0
    best_eval_reward = -float('inf')

    time_step = env.reset()
    current_obs_np = preprocess_observation(time_step.observation)

    progress_bar = tqdm(total=total_timesteps, desc="Training Progress")

    while global_step_count < total_timesteps:
        agent.actor.eval() 
        agent.critic.eval() 
        
        for step_in_rollout_batch in range(rollout_steps_N):
            progress_bar.update(1)
            global_step_count += 1

            with torch.no_grad():
                obs_tensor = torch.as_tensor(current_obs_np, dtype=torch.float32).to(device)
                scaled_action_tensor, log_prob_tensor, _, action_pre_tanh_tensor = agent.actor.get_action_log_prob(obs_tensor)
                value_tensor = agent.critic(obs_tensor).squeeze(-1)

            action_to_env = scaled_action_tensor.cpu().numpy().flatten()
            time_step = env.step(action_to_env) 
            
            next_obs_np = preprocess_observation(time_step.observation)
            reward = time_step.reward if time_step.reward is not None else 0.0
            done = time_step.last()

            rollout_buffer.add(
                obs=current_obs_np,
                action=action_pre_tanh_tensor.cpu().numpy().flatten(),
                reward=reward,
                done=done,
                value=value_tensor.cpu().item(),
                log_prob=log_prob_tensor.cpu().item()
            )
            current_obs_np = next_obs_np

            if done or (step_in_rollout_batch + 1) % max_episode_len == 0 : 
                time_step = env.reset()
                current_obs_np = preprocess_observation(time_step.observation)
            
            if rollout_buffer.is_full() or global_step_count >= total_timesteps:
                break 
            
        # PPO Update
        bootstrap_value = torch.tensor(0.0, device=device, dtype=torch.float32)
        if not done:
            with torch.no_grad():
                obs_for_bootstrap_tensor = torch.as_tensor(current_obs_np, dtype=torch.float32).unsqueeze(0).to(device)
                bootstrap_value = agent.critic(obs_for_bootstrap_tensor).squeeze() 
        
        data_from_buffer = rollout_buffer.get_all_data()
        with torch.no_grad():
            advantages, returns_target = compute_gae_advantages(
                data_from_buffer['rewards'],
                data_from_buffer['values'],
                data_from_buffer['dones'],
                agent.gamma,
                agent.gae_lambda,
                bootstrap_value
            )
            
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        def update_with_bootstrap(self_agent, bootstrap_val): 
            original_update = self_agent.update
            metrics = agent.update_with_precomputed_advantages(
                data_from_buffer['observations'],
                data_from_buffer['actions'],
                data_from_buffer['log_probs'],
                advantages,
                returns_target
            )
            
        metrics = agent.update(
            data_from_buffer['observations'],
            data_from_buffer['actions'],
            data_from_buffer['log_probs'],
            advantages,
            returns_target
        )

        rollout_buffer.reset()
        ppo_update_count += 1

        if ppo_update_count % log_interval_updates == 0 and writer:
            print(f"Global Steps: {global_step_count}, PPO Updates: {ppo_update_count}, uth_t Loss: {metrics['actor_loss']:.4f}, Critic Loss: {metrics['critic_loss']:.4f}, Entropy: {metrics['entropy']:.4f}")
            writer.add_scalar("Loss/uth_t", metrics['actor_loss'], global_step_count)
            writer.add_scalar("Loss/Critic", metrics['critic_loss'], global_step_count)
            writer.add_scalar("Parameters/Entropy", metrics['entropy'], global_step_count)
            writer.add_scalar("Parameters/LR_uth_t", agent.optimizer_actor.param_groups[0]['lr'], global_step_count)
            writer.add_scalar("Parameters/LR_Critic", agent.optimizer_critic.param_groups[0]['lr'], global_step_count)

            if hasattr(agent.actor, 'log_std'):
                 writer.add_scalar("Parameters/Policy_Std_Avg", torch.exp(agent.actor.log_std).mean().item(), global_step_count)


        if ppo_update_count % save_interval_updates == 0:
            save_checkpoint({
                'global_step': global_step_count,
                'ppo_update': ppo_update_count,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
                'best_eval_reward': best_eval_reward
            }, checkpoint_dir=checkpoint_dir, filename=f"ckpt_update_{ppo_update_count}.pth.tar", experiment_name=experiment_name)

        if global_step_count // eval_interval_steps > (global_step_count - rollout_steps_N) // eval_interval_steps:
            eval_reward = run_evaluation_episode(env, actor, preprocess_observation, device, max_steps=max_episode_len)
            print(f"Evaluation at {global_step_count} steps: Reward = {eval_reward:.2f}")
            if writer:
                writer.add_scalar("Evaluation/EpisodeReward", eval_reward, global_step_count)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"New best evaluation reward: {best_eval_reward:.2f}. Saving best model.")
                save_checkpoint({
                    'global_step': global_step_count,
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                }, checkpoint_dir=checkpoint_dir, filename="model_best.pth.tar", is_best=False, experiment_name=experiment_name)

    progress_bar.close()
    env.close()
    if writer:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_walker.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)