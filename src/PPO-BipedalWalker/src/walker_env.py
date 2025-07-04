# walker_ppo/src/envs/walker_env.py
from dm_control import suite
import numpy as np
from typing import Any, Dict, Optional
from .reward_wrappers import WalkerRewardWrapper

def preprocess_observation(dm_observation_dict: Dict[str, Any]) -> np.ndarray:
    orientations = np.asarray(dm_observation_dict['orientations']).ravel()
    height = np.asarray([dm_observation_dict['height']]).ravel()
    velocity = np.asarray(dm_observation_dict['velocity']).ravel()
    return np.concatenate([orientations, height, velocity])

def create_walker_env(seed: int = 42, 
                      task_name: str = 'walk', 
                      reward_wrapper_config: Optional[Dict[str, Any]] = None):
    
    task_kwargs = {'random': np.random.RandomState(seed)}
    base_env = suite.load(domain_name="walker", task_name=task_name, task_kwargs=task_kwargs)
    
    env_to_use = base_env 
   
    if reward_wrapper_config is None:
        reward_wrapper_config = {}
        
    if reward_wrapper_config.get("use_custom_wrapper", False):
        print("Applying custom WalkerRewardWrapper.")
        
        wrapper_kwargs = {
            key: value for key, value in reward_wrapper_config.items() 
            if key != "use_custom_wrapper"
        }

        if task_name == 'run' and 'target_forward_speed' not in wrapper_kwargs:
            print(f"Task is 'run' and target_forward_speed not in config. Setting to 8.0 for wrapper.")
            wrapper_kwargs['target_forward_speed'] = 8.0
            wrapper_kwargs.setdefault('max_rewarded_speed', 8.0)
        
        env_to_use = WalkerRewardWrapper(
            base_env,
            **wrapper_kwargs 
        )
    else:
        print("Using default dm_control Walker environment reward (custom wrapper not enabled in config).")

    time_step = env_to_use.reset() 
    processed_obs_sample = preprocess_observation(time_step.observation)
    observation_dim = processed_obs_sample.shape[0]
    
    action_dim = env_to_use.action_spec().shape[0]
    
    return env_to_use, observation_dim, action_dim