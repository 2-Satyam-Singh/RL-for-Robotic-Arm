# example/control_random_joint.py
"""
This is the example file to quickly test the RobotController class with random joint actions.
It demonstrates how to get joint states, set random joint positions, and retrieve the end-effector pose after actions. 
The code includes comments to guide you through each step of the process.
"""
import argparse
import random
import time

# Ensure you run this from root: python -m example.control_random_joint
from config import ROBOT_CONFIGS
from controller.robot_controller import RobotController

random.seed(42)
LOOP_FREQUENCY = 1

def main():
    parser = argparse.ArgumentParser(description="Test RobotController with random joint actions.")
    parser.add_argument("--robot", choices=["panda", "3dof", "5dof"], default="panda")
    args = parser.parse_args()

    # Load configuration
    cfg = ROBOT_CONFIGS[args.robot]
    joints = cfg["joints"]
    limits = cfg["limits"]

    # Initialize controller with config parameters
    ctrl = RobotController(
        joint_names=joints,
        model_name=cfg["model_name"],
        world_name=cfg["world_name"],
        ee_link_name=cfg["ee_link_name"],
        entities=cfg["entities"]
    )
    
    print(f"[init] Controller active for {args.robot.upper()}.")
    # Brief delay to allow initial messages to arrive
    time.sleep(0.5)

    i = 0
    while True:

# >---> Reset
        # if i % 10 == 0:
        #     print("[reset] Resetting the world.")
        #     ctrl.reset()
        #
        #     # >---> Update Entities after Reset
        #     entities = ctrl.get_entity_positions()
        #     print(f"[state] Positions: {entities}\n")
        #
        #     for name, entity_pos in entities.items():
        #         new_pos = [entity_pos[0] + 1.0, entity_pos[1] + 1.0, entity_pos[2] + 1.0]
        #         ctrl.set_entity_positions(name, new_pos)
        #
        #     entities = ctrl.get_entity_positions()
        #     print(f"[state] Positions: {entities}\n")
        #
        #     time.sleep(LOOP_FREQUENCY)

# >---> Previous State
        js = ctrl.get_joint_states()  # Get joint states
        print(f"[state] Joints: {js}")  # Print old joint states

        entities_dict = ctrl.get_entity_positions()  # Print latest entity positions
        print(f"[state] Positions: {entities_dict}\n")

# >---> Action Performed
        # Generate random actions based on specific robot limits
        action_pos = {j: random.uniform(*lim) for j, lim in zip(joints, limits)}
        
        # OR
        # action_pos = {j: js[j] + 1 if js[j] is not None else random.uniform(*lim) for j, lim in zip(joints, limits)}

        ctrl.set_joint_positions(action_pos)
        time.sleep(LOOP_FREQUENCY)       # Need this time to move the joints

# >---> New State & Result (NEW: Add EE pose here)
        # print(f"[state] Joints: {ctrl.get_joint_states()}")  # Print new joint states
        # NEW: Get and print EE pose after action (shows effect on end-effector)
        ee_pose = ctrl.get_end_effector_pose()
        if ee_pose:
            # Round EE pose to 6 decimal places for printing
            ee_pos = ee_pose['position']
            ori = ee_pose['orientation']
            print(f"[state] EE Pose: Pos={ee_pos}, Ori (quaternion)= {ori}")  # Full quaternion
        else:
            print("[state] EE Pose: No data yet")

        i += 1

if __name__ == "__main__":
    main()