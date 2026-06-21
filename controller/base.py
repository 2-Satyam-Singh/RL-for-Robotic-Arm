# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

# controller/base.py
"""
RobotBackend — the simulator / embodiment seam.

The environment talks to the simulated (or real) robot ONLY through this small
interface. Gazebo is one implementation (`RobotController` in robot_controller.py);
`FakeController` (fake.py) is another, used for headless tests. A future
`MujocoController` / `PyBulletController` / real-robot driver just implements the
same methods and the environment does not change.

This is a `typing.Protocol`: backends conform *structurally* (no inheritance
required), which keeps the seam loose enough to also wrap third-party sim envs
later. The env duck-types these calls; the optional ones below are guarded with
`hasattr`, so a backend may omit them.

Dict shapes (the contract):
    get_joint_states()    -> {joint_name: position_float_or_None}
    get_entity_positions()-> {entity_name: [x, y, z]}
    set_joint_positions(positions: {joint_name: target_position})
    get_end_effector_pose() (optional) -> {"position": [x,y,z],
                                           "orientation": [x,y,z,w]} | None
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    # --- required: observation side ---
    def get_joint_states(self) -> dict:
        """Return {joint_name: position_or_None} for every actuated joint."""
        ...

    def get_entity_positions(self) -> dict:
        """Return {entity_name: [x, y, z]} for tracked world objects."""
        ...

    # --- required: action side ---
    def set_joint_positions(self, positions: dict) -> None:
        """Command joint targets: {joint_name: target_position} (raw units)."""
        ...

    # --- required: episode lifecycle ---
    def reset(self) -> None:
        """Reset the world to a (possibly randomized) initial state."""
        ...

    # --- optional (env guards these with hasattr) ---
    def get_end_effector_pose(self) -> "dict | None":
        """Optional: {'position': [x,y,z], 'orientation': [x,y,z,w]} or None."""
        ...

    def close(self) -> None:
        """Optional: release resources (nodes, sockets, sim handles)."""
        ...

    # NOTE (future, intentionally not part of Phase 1):
    #   def advance(self, dt: float) -> None
    #       Step the simulation forward by `dt`. Today the env does
    #       `time.sleep(dt)` because Gazebo runs asynchronously in its own
    #       process; a MuJoCo/PyBullet backend would instead `sim.step()` here.
    #       When the second backend lands, move that wall-clock sleep out of the
    #       env and behind this method so the env stops assuming a real-time,
    #       asynchronous simulator.
