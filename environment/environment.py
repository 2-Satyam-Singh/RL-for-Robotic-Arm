# environment/environment_0_4.py
"""
Environment wrapper around PandaController.
Exposes a Gym-like API: reset(), step(action).
Normalizes + flattens observations for learning algorithms.
"""

import time
import numpy as np
from controller.panda_controller import PandaController
from reward.reward import compute_reward, reset_reward


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

    def _normalize_obs(self, obs):
        """
        Convert dict observation -> flat normalized vector.
        - Joint angles normalized by joint limits to [-1, 1].
        - Entity positions normalized by workspace_range to [-1, 1].
        """
        # Validate observation
        for j in self.joints:
            if j not in obs["joints"] or obs["joints"][j] is None:
                print(f"[Warning] Joint {j} missing or None in observation, using 0.0")
                obs["joints"][j] = 0.0
        for e in self.entities:
            if e not in obs["entities"] or obs["entities"][e] is None:
                print(f"[Warning] Entity {e} missing or None in observation, using [0, 0, 0]")
                obs["entities"][e] = [0.0, 0.0, 0.0]

        # Normalize joints
        joint_vals = []
        for (j, (low, high)) in zip(self.joints, self.limits):
            val = obs["joints"].get(j, 0.0)
            if high > low:
                norm_val = 2.0 * (val - low) / (high - low) - 1.0  # scale to [-1, 1]
            else:
                norm_val = 0.0
            joint_vals.append(norm_val)

        # Normalize entities
        entity_vals = []
        for e in self.entities:
            pos = obs["entities"].get(e, [0.0, 0.0, 0.0])
            entity_vals.extend([max(-1.0, min(1.0, p / self.workspace_range)) for p in pos])

        flat_obs = np.array(joint_vals + entity_vals, dtype=np.float32)
        return flat_obs

    def _action_to_dict(self, action):                      # is this even necessary?
        """
        Convert action (array or dict) to dict expected by PandaController.
        If array, assume same order as self.joints.
        """
        if isinstance(action, dict):
            return {j: action.get(j, 0.0) for j in self.joints}
        elif isinstance(action, (list, np.ndarray)):
            if len(action) != len(self.joints):
                raise ValueError(f"Action array length {len(action)} does not match joints {len(self.joints)}")
            return dict(zip(self.joints, action))
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def reset(self):
        """Reset robot + reward, return initial normalized state."""
        self.ctrl.reset()
        reset_reward()
        time.sleep(self.dt)
        self.step_count = 0
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
        }
        return self._normalize_obs(obs)

    def step(self, action):                                 # The newly introduced end effector pose (position and orientation) also needs to be used for reward calculation, so we need to add it to the observation space
        """
        Apply action, step sim, return (state, reward, done, info).
        Action can be a NumPy array or dict.
        """
        # Convert action to dict for PandaController
        action_dict = self._action_to_dict(action)
        self.ctrl.set_joint_positions(action_dict)
        time.sleep(self.dt)

        # Collect new state
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
        }
        state = self._normalize_obs(obs)

        # Reward + termination
        reward = compute_reward(obs, self.entities, self.limits)
        self.step_count += 1
        done = self.step_count >= self.max_steps or reward > 0.9

        # Include raw observation in info for debugging
        info = {"raw_obs": obs}

        return state, reward, done, info
