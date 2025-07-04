import torch
import numpy as np
from dm_control import suite
from dm_control import viewer

from src.models.policy import uth_t
from src.envs.walker_env import preprocess_observation
from src.utils import set_device
import yaml

def load_trained_actor(checkpoint_path, obs_dim, action_dim, actor_config, device):
    actor = uth_t(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hdim=actor_config.get('hdim', 256),
        log_std_init=actor_config.get('log_std_init', -0.5)
    ).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'actor_state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
        elif 'model_state_dict' in checkpoint:
             actor.load_state_dict(checkpoint['model_state_dict'])
        else:
            actor.load_state_dict(checkpoint)
        print(f"uth_t checkpoint loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Ensure the checkpoint_path and keys are correct.")
        raise
        
    actor.eval()
    return actor

def main_visualize_dm_control():
    config = {
        "env_seed": 42,
        "obs_dim": 24,
        "action_dim": 6,
        "actor_config": {
            "hdim": 256,
            "log_std_init": -0.5
        },
        "checkpoint_path": "checkpoints/ppo_walker_experiment/model_best.pth.tar", 
        "device": "auto"
    }

    device = set_device(config["device"])
    env = suite.load(
        domain_name="walker", 
        task_name="walk", 
        task_kwargs={'random': np.random.RandomState(config["env_seed"])}
    )

    actor_network = load_trained_actor(
        config["checkpoint_path"],
        config["obs_dim"],
        config["action_dim"],
        config["actor_config"],
        device
    )

    
    def rendering_policy(time_step):
        current_physics_time = env.physics.time()
        print(f"Policy Called | Physics Time: {current_physics_time:.4f} | Step Type: {time_step.step_type}")
        if time_step.observation:
            height = time_step.observation.get('height', 'N/A')
            orientation_sample = time_step.observation.get('orientations', [np.nan]*14)[0] 
            print(f"Obs (Height): {height}, Obs (Orient[0]): {orientation_sample:.4f} | Reward: {time_step.reward}")

        if time_step.first():
            pass 

        processed_obs_np = preprocess_observation(time_step.observation)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(processed_obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            scaled_action_tensor, _, _, _ = actor_network.get_action_log_prob(obs_tensor, deterministic=True)

        action_to_env = scaled_action_tensor.cpu().numpy().flatten()
        print(f"Action: {action_to_env}")
        return action_to_env


    print("Launching dm_control viewer... Close the viewer window to stop.")
    viewer.launch(env, policy=rendering_policy)

if __name__ == '__main__':
    main_visualize_dm_control()