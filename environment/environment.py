# environment/environment.py
"""
RobotEnv — a Gymnasium environment wrapping a RobotBackend (Gazebo by default).

Standard Gymnasium API:
    reset(seed, options) -> (obs, info)
    step(action)         -> (obs, reward, terminated, truncated, info)

Observations are normalized + flattened to a Box in [-1, 1]; actions are a Box in
[-1, 1] (normalized joint targets, denormalized to joint limits before actuation).

Layers (see controller/base.py):
    embodiment / backend = robot + simulator  -> self.ctrl : RobotBackend
    task                 = reward + success + reset (today: sparse "knock off table")
This class composes one backend + one task into a gymnasium.Env.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import ROBOT_CONFIGS
from reward.sparse_reward import compute_reward, reset_reward
from utils.env_utils import normalize_obs, denormalize_action, action_to_dict

GROUND_Z = 0.10   # object counts as "knocked off" once its z drops to/below this


class RobotEnv(gym.Env):
    """Gymnasium env for a joint-controlled robot in Gazebo (or any RobotBackend)."""

    metadata = {"render_modes": []}

    def __init__(self, joints, limits, entities, workspace_range=0.85, max_steps=100,
                 model_name="panda", world_name="panda_world", ee_link_name="panda_hand",
                 reward_type="dense", loop_hz=50, controller=None, render_mode=None):

        self.joints = joints
        self.limits = limits
        self.entities = entities
        self.workspace_range = workspace_range
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.dt = 1.0 / loop_hz
        self.step_count = 0
        self.render_mode = render_mode

        # --- embodiment/backend seam: inject any RobotBackend; default = Gazebo ---
        if controller is not None:
            self.ctrl = controller
        else:
            # Lazy import so merely importing RobotEnv (e.g. with a Fake/MuJoCo
            # backend) never requires the Gazebo (gz-transport) libraries.
            from controller.robot_controller import RobotController
            self.ctrl = RobotController(
                joint_names=joints,
                model_name=model_name,
                world_name=world_name,
                ee_link_name=ee_link_name,
                entities=entities,
            )

        # --- Gymnasium spaces (single source of truth for obs/action dims) ---
        n_joints = len(joints)
        n_entities = len(entities)
        obs_dim = n_joints + 3 * n_entities
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_joints,), dtype=np.float32)

    def _get_obs_dict(self):
        return {
            "joints": self.ctrl.get_joint_states(),
            "entities": self.ctrl.get_entity_positions(),
            # "ee_pose": self.ctrl.get_end_effector_pose()   # Phase-2: collect EE pose
        }

    def reset(self, *, seed=None, options=None):
        """Reset robot + reward; return (normalized_obs, info)."""
        super().reset(seed=seed)
        self.ctrl.reset()
        reset_reward()
        self.step_count = 0

        obs = self._get_obs_dict()
        state = normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)
        info = {"raw_obs": obs}
        return state, info

    def step(self, action):
        """Apply action, step sim, return (obs, reward, terminated, truncated, info)."""
        if not isinstance(action, dict):
            action = denormalize_action(action, self.joints, self.limits)
            action_dict = action_to_dict(action, self.joints)
        else:
            action_dict = action

        self.ctrl.set_joint_positions(action_dict)
        time.sleep(self.dt)

        obs = self._get_obs_dict()
        state = normalize_obs(obs, self.joints, self.limits, self.entities, self.workspace_range)

        # Reward (task layer). NOTE: reward_type is not yet wired — see deferred #8.
        reward = compute_reward(obs, self.entities, self.limits)

        self.step_count += 1

        # --- terminated (task success) vs truncated (time limit) ---
        target_name = next((name for name in obs["entities"] if name in self.entities), None)
        curr_z = (obs["entities"][target_name][2]
                  if target_name and len(obs["entities"][target_name]) > 2 else 999)
        terminated = bool(curr_z <= GROUND_Z)
        truncated = bool(self.step_count >= self.max_steps)

        info = {"raw_obs": obs, "is_success": terminated}
        return state, float(reward), terminated, truncated, info

    def render(self):
        """No-op: visualization is handled by the external Gazebo GUI."""
        return None

    def close(self):
        if hasattr(self.ctrl, "close"):
            self.ctrl.close()


# --- Register one Gymnasium env ID per robot config: Robot-<name>-v0 ---
# (Importing this module performs the registration, so `gymnasium.make("Robot-panda-v0")`
#  works for any code that has imported the env at least once.)
for _name, _cfg in ROBOT_CONFIGS.items():
    _env_id = f"Robot-{_name}-v0"
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point="environment.environment:RobotEnv",
            kwargs={
                "joints": _cfg["joints"],
                "limits": _cfg["limits"],
                "entities": _cfg["entities"],
                "workspace_range": _cfg["workspace_range"],
                "model_name": _cfg["model_name"],
                "world_name": _cfg["world_name"],
                "ee_link_name": _cfg["ee_link_name"],
            },
        )
