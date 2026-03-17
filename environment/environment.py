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
from controller.panda_controller import PandaController, JOINTS, LIMITS
from reward.reward import compute_reward, reset_reward
from utils.env_utils import normalize_obs, denormalize_action, action_to_dict


class PandaEnv:
    def __init__(self, joints, limits, entities, loop_hz=10, workspace_range=0.85):
        """
        Args:
            joints: list of joint names
            limits: list of (low, high) tuples for each joint
            entities: set/list of entity names to track
            loop_hz: control frequency
            workspace_range: max workspace distance in meters (≈ robot reach)
        """
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.ctrl = PandaController(joints)
        self.ctrl.start_pose_monitoring()
        self.dt = 1.0 / loop_hz
        self.step_count = 0
        self.max_steps = 200
        self.workspace_range = workspace_range

    def reset(self):
        """Reset robot + reward, return initial normalized state."""
        pos = self.ctrl.get_entity_positions()  # Print latest entity positions
        print(f"[state] Positions: {pos}\n")
        self.ctrl.reset()
        reset_reward()
        time.sleep(self.dt)
        self.step_count = 0
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
            # "ee_pose": self.ctrl.get_end_effector_pose()    #Include this in future

        }
        return normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)

    def step(self, action):                                 # The newly introduced end effector pose (position and orientation) also needs to be used for reward calculation, so we need to add it to the observation space
        """
        Apply action, step sim, return (state, reward, done, info).
        Action can be a NumPy array or dict.
        """
        # Convert action to dict for PandaController
        action_dict = action_to_dict(action, self.joints)
        self.ctrl.set_joint_positions(action_dict)
        time.sleep(self.dt)

        # Collect new state
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
            # "ee_pose": self.ctrl.get_end_effector_pose()    #Include this in future, but it would need major changes
        }
        state = normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)

        # Reward + termination
        reward = compute_reward(obs, self.entities, self.limits)
        self.step_count += 1
        done = self.step_count >= self.max_steps or reward > 0.9

        # Include raw observation in info for debugging
        info = {"raw_obs": obs}

        return state, reward, done, info