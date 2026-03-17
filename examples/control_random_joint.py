"""
This is the main file and it currently perform 
JOINT VALUE UPDATIONS
RESET the environment
RECIVENG JOINT STATE
Updating Entities Position
"""
import random
import time
from controller.panda_controller import PandaController, JOINTS, LIMITS

random.seed(42)

# ──────── Removed finger joints for training (only 7 arm joints now) ────────
JOINTS = JOINTS[:-2]      # panda_joint1..7 only
LIMITS = LIMITS[:-2]

LOOP_FREQUENCY = 1

def main():
    ctrl = PandaController(JOINTS)
    print("[init] Panda controller active.")
    # Brief delay to allow initial messages to arrive
    time.sleep(0.5)

    i = 0
    while True:

# >---> Reset
        # if i % 10 == 0:
        #     print("[reset] Resetting the world.")
        #     ctrl.reset()

        #     # >---> Update Entities after Reset
        #     entities = ctrl.get_entity_positions()
        #     print(f"[state] Positions: {entities}\n")

        #     for name, pos in entities.items():
        #         new_pos = [pos[0] + 1.0, pos[1] + 1.0, pos[2] + 1.0]
        #         ctrl.set_entity_positions(name, new_pos)


        #     entities = ctrl.get_entity_positions()
        #     print(f"[state] Positions: {entities}\n")

        #     time.sleep(LOOP_FREQUENCY)



# >---> Previous State
        js = ctrl.get_joint_states()  # Get joint states
        print(f"[state] Joints: {js}")  # Print old joint states

        pos = ctrl.get_entity_positions()  # Print latest entity positions
        print(f"[state] Positions: {pos}\n")


# >---> Action Performed
        pos = {j: random.uniform(*lim) for j, lim in zip(JOINTS, LIMITS)}  # Random actions
        # OR
        # pos = {j: js[j] + 1 if js[j] is not None else random.uniform(*lim) for j, lim in zip(JOINTS, LIMITS)}  # Handle None safely if uncommented

        ctrl.set_joint_positions(pos)
        time.sleep(LOOP_FREQUENCY)       # Need this time to move the joints


# >---> New State & Result (NEW: Add EE pose here)
        # print(f"[state] Joints: {ctrl.get_joint_states()}")  # Print new joint states
        # NEW: Get and print EE pose after action (shows effect on end-effector)
        ee_pose = ctrl.get_end_effector_pose()
        if ee_pose:
            # Round EE pose to 6 decimal places for printing
            pos = ee_pose['position']
            ori = ee_pose['orientation']
            print(f"[state] EE Pose: Pos={pos}, Ori (quaternion)= {ori}")  # Full quaternion
        else:
            print("[state] EE Pose: No data yet")

        i += 1

if __name__ == "__main__":
    main()