# algorithms/ml_policy.py
"""
MLPolicyAgent — runs a *distilled* scikit-learn / XGBoost policy as a drop-in
replacement for the RL agents (PPOAgent / DQNAgent) during evaluation.

The distilled models were trained (see Gazebo_sim.ipynb) to imitate the PPO
policy. Instead of mapping the raw observation -> action, they map a set of
engineered kinematic features -> the *delta* of the (normalized) action:

        delta_action_t = action_t - action_{t-1}

so the executed action is reconstructed as `action_t = action_{t-1} + delta`.

This class mirrors the minimal agent API used by scripts/test.py:
    algo = MLPolicyAgent(env, model_type="RF")
    algo.load("7dof.pkl")          # loads models/RF/7dof.pkl + models/SCALER/7dof.pkl
    algo.reset_episode()           # call at the start of every episode
    action = algo.select_action(obs)

Feature layout (must match the saved StandardScaler.feature_names_in_):
    [ee_error_x, ee_error_y, ee_error_z, ee_dist, progress]
    + per joint i:
      [sin_joint_i, cos_joint_i, vel_i, accel_i, jerk_i,
       vel_accel_i, prev_action_i, prev_action_2_i, prev_vel_i]

vel/accel/jerk are TIME-derivatives in the notebook:
    vel = d(joint)/dt,  accel = d(vel)/dt,  jerk = d(accel)/dt
where dt = step_time.diff() floored at 1e-5. step_time is per-step wall-clock
latency, so dt is (a) non-causal — dt_t needs step_time_t, which is only known
*after* action_t is chosen — and (b) machine/run dependent, hence not
reproducible at inference. We therefore use a fixed dt = DT_INFERENCE (the
notebook's 1e-5 floor, which dominates the training data): this keeps the
feature magnitudes in the range the scaler expects, makes them track real joint
motion, and keeps the policy fully deterministic for reproducible results.
"""

import os
import numpy as np
import joblib

# Root folder that holds the per-model-type sub-directories (LR/ RF/ SVM/ XGBoost/ SCALER/)
MODEL_ROOT = "models"

# Goal/target reference used when the distillation features were engineered.
# ee_error_* = entity_position - GOAL_XYZ   (hard-coded in Gazebo_sim.ipynb)
GOAL_XYZ = np.array([0.5, 0.0, 0.5], dtype=float)

# Fixed time-delta used to turn finite differences into velocity/accel/jerk.
# Equals the `dt_safe` floor used during training (Gazebo_sim.ipynb, CELL 2C).
DT_INFERENCE = 1e-5

# Accepted spellings for the model-type CLI argument -> on-disk directory name.
MODEL_TYPE_ALIASES = {
    "lr": "LR", "linear": "LR", "linearregression": "LR",
    "rf": "RF", "randomforest": "RF", "random_forest": "RF",
    "svm": "SVM", "svr": "SVM",
    "xgb": "XGBoost", "xgboost": "XGBoost",
}


def resolve_model_type(name):
    """Map a user-supplied model-type string to its on-disk directory name."""
    key = str(name).strip().lower()
    if key in MODEL_TYPE_ALIASES:
        return MODEL_TYPE_ALIASES[key]
    # Fall back to matching a directory name directly (case-insensitive).
    for d in ("LR", "RF", "SVM", "XGBoost"):
        if key == d.lower():
            return d
    raise ValueError(
        f"Unknown ML model type '{name}'. "
        f"Choose one of: LR, RF, SVM, XGBoost."
    )


class MLPolicyAgent:
    """A distilled supervised-learning model wrapped as an evaluation policy."""

    def __init__(self, env, model_type="RF", model_root=MODEL_ROOT):
        self.model_type = resolve_model_type(model_type)
        self.model_root = model_root

        self.joints = list(env.joints)
        self.limits = np.asarray(env.limits, dtype=float)
        self.entities = sorted(list(env.entities)) if env.entities else []
        self.workspace_range = getattr(env, "workspace_range", 0.85)
        self.n_joints = len(self.joints)
        self.max_steps = int(getattr(env, "max_steps", 100))

        self.model = None
        self.scaler = None
        self.model_path = None

        self.reset_episode()

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def load(self, name):
        """Load the regressor + matching scaler.

        `name` is the short ML filename, e.g. "7dof", "7dof.pkl" or a full path;
        only the basename is used. The scaler is taken from models/SCALER/<file>.
        """
        fname = os.path.basename(str(name))
        if not fname.endswith(".pkl"):
            fname += ".pkl"

        self.model_path = os.path.join(self.model_root, self.model_type, fname)
        scaler_path = os.path.join(self.model_root, "SCALER", fname)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ML model not found: {self.model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(scaler_path)

        # Guard against running a model against the wrong robot (DOF mismatch).
        expected = 9 * self.n_joints + 5
        got = getattr(self.scaler, "n_features_in_", expected)
        if got != expected:
            raise ValueError(
                f"Feature mismatch: '{fname}' expects {got} features but the "
                f"'{'/'.join(self.joints[:1]) or 'robot'}' env ({self.n_joints} DOF) "
                f"produces {expected}. Did you pair the right --robot with --model_name?"
            )

    # ------------------------------------------------------------------
    # EPISODE STATE
    # ------------------------------------------------------------------
    def reset_episode(self):
        """Clear the per-episode recurrence. Call once before each episode."""
        self.prev_joint = None                       # joint angles at step t-1 (rad)
        self.prev_vel = np.zeros(self.n_joints)      # velocity at step t-1
        self.prev_accel = np.zeros(self.n_joints)    # acceleration at step t-1
        self.prev_action = np.zeros(self.n_joints)   # executed action at step t-1
        self.prev_action_2 = np.zeros(self.n_joints)  # executed action at step t-2
        self.t = 0                                   # number of steps taken this episode

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING (streaming equivalent of the notebook's pandas pipeline)
    # ------------------------------------------------------------------
    def _denorm_joints(self, obs):
        """Normalized [-1,1] joints -> radians (matches Logger.flatten_state_action)."""
        jn = np.asarray(obs[:self.n_joints], dtype=float)
        lows, highs = self.limits[:, 0], self.limits[:, 1]
        return np.where(highs > lows, lows + (jn + 1.0) * (highs - lows) / 2.0, lows)

    def _denorm_entity(self, obs):
        """First entity's normalized xyz -> metres (matches Logger.flatten_state_action)."""
        e = np.asarray(obs[self.n_joints:self.n_joints + 3], dtype=float)
        return e * self.workspace_range

    def _build_features(self, joints, ent_xyz):
        """Assemble the 9*DOF+5 feature vector for the current step and advance
        the kinematic recurrence (velocity/acceleration/jerk)."""
        # --- base spatial features ---
        ee_err = ent_xyz - GOAL_XYZ
        ee_dist = float(np.sqrt(np.sum(ee_err ** 2)))
        # progress: episode length is unknown at inference, so max_steps is the
        # only causal denominator available (the notebook used the realized length).
        self.t += 1
        progress = self.t / self.max_steps

        # --- per-joint kinematic features (time-derivatives, see module docstring) ---
        if self.prev_joint is None:
            vel = np.zeros(self.n_joints)            # first row: joint diff is 0 -> vel 0
        else:
            vel = (joints - self.prev_joint) / DT_INFERENCE
        accel = (vel - self.prev_vel) / DT_INFERENCE   # uses last step's (possibly 0) velocity
        jerk = (accel - self.prev_accel) / DT_INFERENCE
        prev_vel_feat = self.prev_vel.copy()
        vel_accel = vel * accel

        feat = [ee_err[0], ee_err[1], ee_err[2], ee_dist, progress]
        for i in range(self.n_joints):
            feat += [
                np.sin(joints[i]), np.cos(joints[i]),
                vel[i], accel[i], jerk[i], vel_accel[i],
                self.prev_action[i], self.prev_action_2[i], prev_vel_feat[i],
            ]

        # advance kinematic state for the next step
        self.prev_joint = joints
        self.prev_vel = vel
        self.prev_accel = accel
        return np.asarray(feat, dtype=float)

    # ------------------------------------------------------------------
    # POLICY
    # ------------------------------------------------------------------
    def select_action(self, obs):
        """Return the (normalized) action array for the given observation."""
        joints = self._denorm_joints(obs)
        ent_xyz = self._denorm_entity(obs)

        feat = self._build_features(joints, ent_xyz)
        scaled = self.scaler.transform(feat.reshape(1, -1))
        delta = np.asarray(self.model.predict(scaled), dtype=float).reshape(-1)[:self.n_joints]

        # reconstruct the absolute action and keep it inside the valid [-1, 1]
        # normalized action range (also keeps prev_action in-distribution).
        action = np.clip(self.prev_action + delta, -1.0, 1.0)

        # advance the action recurrence
        self.prev_action_2 = self.prev_action
        self.prev_action = action
        return action
