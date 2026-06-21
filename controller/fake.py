# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

# controller/fake.py
"""
FakeController — an in-memory `RobotBackend` with no simulator.

Used to verify the Gymnasium contract (`gymnasium.utils.env_checker.check_env`)
and to smoke-test SB3 wiring without launching Gazebo. It is the first alternate
backend behind the `RobotBackend` seam (see controller/base.py).

Deliberately deterministic: a seeded `reset()` is reproducible (joints start at
their limit midpoints, the object sits on the table), and `set_joint_positions`
moves joints instantly to the commanded targets so observations evolve as the
agent acts. There is no physics here — the real dynamics live in Gazebo.
"""
import numpy as np


class FakeController:
    """Minimal deterministic stand-in for RobotController (no gz, no physics)."""

    DECIMAL_PLACES = 2

    def __init__(self, joint_names, entities=None, limits=None,
                 object_pos=(0.5, 0.0, 1.025)):
        self.joint_names = list(joint_names)
        self.entity_names = list(entities) if entities else []
        self.limits = list(limits) if limits is not None else [(-1.0, 1.0)] * len(self.joint_names)
        self._initial_object_pos = [float(p) for p in object_pos]

        self._joints = {}
        self._entities = {}
        self.reset()

    # --- lifecycle ---
    def reset(self):
        """Deterministic initial state: joints at limit midpoints, object on the table."""
        self._joints = {
            j: round(0.5 * (lo + hi), self.DECIMAL_PLACES)
            for j, (lo, hi) in zip(self.joint_names, self.limits)
        }
        self._entities = {name: list(self._initial_object_pos) for name in self.entity_names}

    def close(self):
        pass

    # --- observation side ---
    def get_joint_states(self):
        return dict(self._joints)

    def get_entity_positions(self):
        return {name: list(pos) for name, pos in self._entities.items()}

    def get_end_effector_pose(self):
        return {"position": [0.0, 0.0, 0.5], "orientation": [0.0, 0.0, 0.0, 1.0]}

    # --- action side ---
    def set_joint_positions(self, positions):
        """Instantly move commanded joints to their (limit-clipped) targets."""
        for j, val in positions.items():
            if j in self._joints:
                lo, hi = self.limits[self.joint_names.index(j)]
                self._joints[j] = round(float(np.clip(val, lo, hi)), self.DECIMAL_PLACES)
