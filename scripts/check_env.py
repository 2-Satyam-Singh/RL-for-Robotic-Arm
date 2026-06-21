# Copyright (C) 2026 Satyam Singh
# SPDX-License-Identifier: AGPL-3.0-or-later

# scripts/check_env.py
"""
Headless verification of the Gymnasium contract — no Gazebo required.

Builds RobotEnv on a FakeController (an in-memory RobotBackend) and runs
gymnasium's env_checker, then an optional short SB3 SAC smoke-test to prove the
env wires up to Stable-Baselines3.

    python scripts/check_env.py --robot panda
    python scripts/check_env.py --robot panda --sb3
"""
import argparse
import os
import sys

# Make the repo root importable regardless of how this script is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from config import ROBOT_CONFIGS
from controller.fake import FakeController
from environment.environment import RobotEnv   # importing also registers Robot-<name>-v0


def make_fake_env(robot):
    """RobotEnv for `robot`, driven by an in-memory FakeController (no gz, no physics)."""
    cfg = ROBOT_CONFIGS[robot]
    fake = FakeController(
        joint_names=cfg["joints"],
        entities=cfg["entities"],
        limits=cfg["limits"],
    )
    return RobotEnv(
        joints=cfg["joints"],
        limits=cfg["limits"],
        entities=cfg["entities"],
        workspace_range=cfg["workspace_range"],
        model_name=cfg["model_name"],
        world_name=cfg["world_name"],
        ee_link_name=cfg["ee_link_name"],
        controller=fake,
    )


def main():
    parser = argparse.ArgumentParser(description="Headless Gymnasium contract check (no Gazebo).")
    parser.add_argument("--robot", choices=list(ROBOT_CONFIGS.keys()), default="panda")
    parser.add_argument("--sb3", action="store_true", help="also run a short SB3 SAC smoke-test")
    args = parser.parse_args()

    print(f"[check_env] Building RobotEnv('{args.robot}') on FakeController...")
    env = make_fake_env(args.robot)

    print("[check_env] Running gymnasium.utils.env_checker.check_env ...")
    check_env(env, skip_render_check=True)
    print("[check_env] PASS: env_checker")

    # Registration sanity: every robot config should have produced an env ID.
    ids = [f"Robot-{r}-v0" for r in ROBOT_CONFIGS]
    missing = [i for i in ids if i not in gym.registry]
    assert not missing, f"Unregistered env IDs: {missing}"
    print(f"[check_env] PASS: registered {', '.join(ids)}")

    if args.sb3:
        from stable_baselines3 import SAC
        print("[check_env] Running SB3 SAC smoke-test (200 steps) on the fake backend...")
        model = SAC("MlpPolicy", make_fake_env(args.robot), verbose=0)
        model.learn(total_timesteps=200)
        print("[check_env] PASS: SB3 SAC smoke-test")

    env.close()
    print("[check_env] Done.")


if __name__ == "__main__":
    main()
