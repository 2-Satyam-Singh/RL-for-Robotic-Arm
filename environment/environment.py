# environment/environment_0_3

"""
Environment wrapper around PandaController.
Exposes a Gym-like API: reset(), step(action).
These functions are available:
- reset(): Reset the environment to initial state.
- step(action): Apply action, return (obs, reward, done, info).
"""

import time
from controller.panda_controller import PandaController
from reward.reward import compute_reward, reset_reward

class PandaEnv:
    def __init__(self, joints, limits, entities, loop_hz=10):
        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.ctrl = PandaController(joints)
        self.ctrl.start_pose_monitoring()
        self.dt = 1.0 / loop_hz
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        self.ctrl.reset()
        reset_reward()
        time.sleep(self.dt)
        self.step_count = 0
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
        }
        return obs

    def step(self, action):
        # Apply action
        self.ctrl.set_joint_positions(action)
        time.sleep(self.dt)

        # Collect new state
        obs = {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions()
        }

        # Reward + termination
        reward = compute_reward(obs, self.entities)
        self.step_count += 1
        done = self.step_count >= self.max_steps or reward > 0.9

        return obs, reward, done, {}
