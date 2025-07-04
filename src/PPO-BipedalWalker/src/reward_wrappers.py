# walker_ppo/src/envs/reward_wrappers.py
import collections
from typing import Any, Dict, Optional
import dm_env
from dm_env import specs 
from dm_control.rl import control 
from dm_control.utils import rewards
import numpy as np

# Default Hyperparameters
DEFAULT_UPRIGHT_TORSO_WEIGHT: float = 0.2
DEFAULT_UPRIGHT_THRESHOLD: float = 0.9
DEFAULT_MIN_ALIVE_HEIGHT_THRESHOLD: float = 0.7
DEFAULT_FALL_PENALTY_VALUE: float = -1.0
DEFAULT_TARGET_WALK_HEIGHT: float = 1.15
DEFAULT_CROUCH_PENALTY_WEIGHT: float = 0.5
DEFAULT_FORWARD_SPEED_WEIGHT: float = 1.0
DEFAULT_TARGET_FORWARD_SPEED: float = 1.0 
DEFAULT_MAX_REWARDED_SPEED: float = 1.0 
DEFAULT_UPRIGHT_SPEED_GATE_THRESHOLD: float = 0.85
DEFAULT_MIN_HEIGHT_FOR_SPEED_REWARD: float = 1.0
DEFAULT_CONTROL_COST_WEIGHT: float = 1e-5
DEFAULT_USE_TOLERANCE_FOR_UPRIGHT: bool = True
DEFAULT_USE_TOLERANCE_FOR_SPEED: bool = True
DEFAULT_SPEED_STILLNESS_MARGIN: float = 0.05

WORLD_UP_VECTOR = np.array([0., 0., 1.])

class WalkerRewardWrapper(dm_env.Environment):
    def __init__(self,
                 env: dm_env.Environment,
                 upright_torso_weight: float = DEFAULT_UPRIGHT_TORSO_WEIGHT,
                 upright_threshold: float = DEFAULT_UPRIGHT_THRESHOLD,
                 min_alive_height_threshold: float = DEFAULT_MIN_ALIVE_HEIGHT_THRESHOLD,
                 fall_penalty_value: float = DEFAULT_FALL_PENALTY_VALUE,
                 target_walk_height: float = DEFAULT_TARGET_WALK_HEIGHT,
                 crouch_penalty_weight: float = DEFAULT_CROUCH_PENALTY_WEIGHT,
                 forward_speed_weight: float = DEFAULT_FORWARD_SPEED_WEIGHT,
                 target_forward_speed: float = DEFAULT_TARGET_FORWARD_SPEED, 
                 max_rewarded_speed: float = DEFAULT_MAX_REWARDED_SPEED,
                 upright_speed_gate_threshold: float = DEFAULT_UPRIGHT_SPEED_GATE_THRESHOLD,
                 min_height_for_speed_reward: float = DEFAULT_MIN_HEIGHT_FOR_SPEED_REWARD,
                 control_cost_weight: float = DEFAULT_CONTROL_COST_WEIGHT,
                 use_tolerance_for_upright: bool = DEFAULT_USE_TOLERANCE_FOR_UPRIGHT,
                 use_tolerance_for_speed: bool = DEFAULT_USE_TOLERANCE_FOR_SPEED,
                 speed_stillness_margin: float = DEFAULT_SPEED_STILLNESS_MARGIN):
        self._env = env
        self._physics: Optional[control.Physics] = None 

        # Store reward parameters
        self.upright_torso_weight = upright_torso_weight
        self.upright_threshold = upright_threshold
        self.min_alive_height_threshold = min_alive_height_threshold
        self.fall_penalty_value = fall_penalty_value
        self.target_walk_height = target_walk_height
        self.crouch_penalty_weight = crouch_penalty_weight
        self.forward_speed_weight = forward_speed_weight
        self.target_forward_speed = target_forward_speed
        self.max_rewarded_speed = max_rewarded_speed
        self.upright_speed_gate_threshold = upright_speed_gate_threshold
        self.min_height_for_speed_reward = min_height_for_speed_reward
        self.control_cost_weight = control_cost_weight
        self.use_tolerance_for_upright = use_tolerance_for_upright
        self.use_tolerance_for_speed = use_tolerance_for_speed

    def _initialize_physics_if_needed(self):
        if self._physics is None:
            if hasattr(self._env, 'physics') and self._env.physics is not None:
                self._physics = self._env.physics

    def _calculate_torso_uprightness(self) -> float:
        self._initialize_physics_if_needed()
        torso_xmat = self._physics.named.data.xmat['torso'].reshape(3, 3)
        torso_z_axis_world = torso_xmat[:, 2]
        return np.dot(torso_z_axis_world, WORLD_UP_VECTOR)

    def _calculate_torso_height(self) -> float:
        self._initialize_physics_if_needed()
        return self._physics.named.data.qpos['rootz']

    def _calculate_forward_speed(self) -> float:
        self._initialize_physics_if_needed()
        return self._physics.named.data.qvel['rootx']

    def _calculate_control_cost(self, action: np.ndarray) -> float:
        return -self.control_cost_weight * np.sum(np.square(action))

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        original_timestep = self._env.step(action)
        self._physics = self._env.physics # Physics state is updated after step

        # Upright Torso Reward
        uprightness_metric = self._calculate_torso_uprightness()
        if self.use_tolerance_for_upright:
            upright_reward = self.upright_torso_weight * rewards.tolerance(
                uprightness_metric,
                bounds=(self.upright_threshold, 1.0),
                margin=self.upright_threshold,
                value_at_margin=0.0,
                sigmoid='linear'
            )
        else:
            upright_reward = self.upright_torso_weight * uprightness_metric

        # Fall and Crouch Penalties
        torso_height = self._calculate_torso_height()
        fall_penalty = 0.0
        if torso_height < self.min_alive_height_threshold:
            fall_penalty = self.fall_penalty_value

        crouch_penalty = 0.0
        if self.min_alive_height_threshold <= torso_height < self.target_walk_height:
            height_deficit = self.target_walk_height - torso_height
            crouch_penalty = -self.crouch_penalty_weight * height_deficit

        # Posture-Gated Forward Velocity Reward
        forward_speed = self._calculate_forward_speed()
        speed_reward = 0.0
        is_sufficiently_upright = uprightness_metric > self.upright_speed_gate_threshold
        is_not_crouching_too_much = torso_height > self.min_height_for_speed_reward

        if is_sufficiently_upright and is_not_crouching_too_much:
            if self.target_forward_speed == 0.0:
                speed_tolerance_margin = getattr(self, 'speed_stillness_margin', 0.05)
                target_vel_reward_comp = rewards.tolerance(
                    abs(forward_speed), 
                    bounds=(0.0, 0.0),  
                    margin=speed_tolerance_margin, 
                    value_at_margin=0.0, 
                    sigmoid='gaussian' 
                )
                speed_reward = self.forward_speed_weight * target_vel_reward_comp 
            else:
                speed_tolerance_margin = self.target_forward_speed 
                target_vel_reward_comp = rewards.tolerance(
                    forward_speed,
                    bounds=(self.target_forward_speed, self.target_forward_speed + 1.0),
                    margin=speed_tolerance_margin,
                    value_at_margin=0.0,
                    sigmoid='linear'
                )
                speed_reward = self.forward_speed_weight * target_vel_reward_comp
        else:
            speed_reward = self.forward_speed_weight * np.clip(forward_speed, 0, self.max_rewarded_speed)

        # Control Cost Penalty
        control_cost = self._calculate_control_cost(action)

        # Reward
        custom_total_reward = float(
            upright_reward +
            speed_reward +
            fall_penalty +
            crouch_penalty +
            control_cost
        )
        
        return dm_env.TimeStep(
            step_type=original_timestep.step_type,
            reward=custom_total_reward,
            discount=original_timestep.discount,
            observation=original_timestep.observation
        )

    def reset(self) -> dm_env.TimeStep:
        time_step = self._env.reset()
        self._physics = self._env.physics 
        return time_step

    def action_spec(self) -> specs.BoundedArray:
        return self._env.action_spec()

    def observation_spec(self) -> Dict[str, specs.Array]: 
        return self._env.observation_spec()

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=np.float32, name='reward')

    def discount_spec(self) -> specs.BoundedArray:
        return self._env.discount_spec()

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()