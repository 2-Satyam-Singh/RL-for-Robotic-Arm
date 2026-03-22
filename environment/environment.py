# environment/environment.py
"""
Environment wrapper around PandaController.
Exposes a Gym-like API: reset(), step(action).
Normalizes + flattens observations for learning algorithms.

These functions are available:
- reset(): Reset the environment to initial state.
- step(action): Apply action, return (obs, reward, done, info).
"""

import time
import numpy as np
from controller.panda_controller import PandaController
from reward.sparse_reward import compute_reward, reset_reward
from utils.env_utils import normalize_obs, denormalize_action, action_to_dict

class PandaEnv:   # kept name so your existing code doesn't break
    def __init__(self, joints, limits, entities, workspace_range=0.85, max_steps=100,
                 model_name="panda", world_name="panda_world", ee_link_name="panda_hand",
                 reward_type="dense", loop_hz=50):
        
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.dt = 1.0 / loop_hz
        self.step_count = 0

        # ← now passes the config to controller using explicit keyword arguments
        self.ctrl = PandaController(
            joint_names=joints, 
            model_name=model_name, 
            world_name=world_name, 
            ee_link_name=ee_link_name, 
            entities=entities
        )
        
        # Note: start_pose_monitoring() is already called inside PandaController.__init__
        # so you don't strictly need to call it again here, but it's safe if you do.

    def reset(self):
        """Reset robot + reward, return initial normalized state."""
        self.ctrl.reset()
        reset_reward()
        self.step_count = 0

        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
            # "ee_pose": self.ctrl.get_end_effector_pose()    # Include this in future
        }
        return normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)

    def step(self, action):                                 
        """
        Apply action, step sim, return (state, reward, done, info).
        Action can be a NumPy array or dict.
        """
        if not isinstance(action, dict):
            action = denormalize_action(action, self.joints, self.limits)  
            action_dict = action_to_dict(action, self.joints)
        else:
            action_dict = action
            
        self.ctrl.set_joint_positions(action_dict)
        time.sleep(self.dt)

        # Collect new state
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
            # "ee_pose": self.ctrl.get_end_effector_pose()    # Include this in future, but it would need major changes
        }
        
        state = normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)

        # Reward + termination
        # Note: You may want to pass self.reward_type to compute_reward in the future!
        reward = compute_reward(obs, self.entities, self.limits)   # ← still same line

        self.step_count += 1
        
        # ←←← REPLACE THE done line with:
        target_name = next((name for name in obs["entities"] if name in self.entities), None)
        curr_z = obs["entities"][target_name][2] if target_name and len(obs["entities"][target_name]) > 2 else 999
        done = self.step_count >= self.max_steps or curr_z <= 0.10   # GROUND_Z hard-coded for zero extra import

        info = {"raw_obs": obs}
        return state, reward, done, info