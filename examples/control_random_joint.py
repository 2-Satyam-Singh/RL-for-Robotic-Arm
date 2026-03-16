"""
This is the main file and it currently perform 
JOINT VALUE UPDATIONS
RESET the environment
RECIVENG JOINT STATE
Updating Entities Position
"""
import random
import time
from controller.panda_controller import PandaController

random.seed(42)

JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
LIMITS = [(-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9), (-3.07, -0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9), (0, 0.04), (0, 0.04)]
LOOP_FREQUENCY = 10

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
        # Round joint states to 6 decimal places for printing
        # js_rounded = {k: round(v, 6) if v is not None else None for k, v in js.items()}
        # print(f"[state] Joints: {js_rounded}")  # Print rounded joint states
        print()

        # pos = ctrl.get_entity_positions()  # Print latest entity positions
        # print(f"[state] Positions: {pos}\n")


# >---> Action Performed
        # pos = {j: random.uniform(*lim) for j, lim in zip(JOINTS, LIMITS)}  # Random actions
        # OR
        # pos = {j: js[j] + 1 if js[j] is not None else random.uniform(*lim) for j, lim in zip(JOINTS, LIMITS)}  # Handle None safely if uncommented

        # ctrl.set_joint_positions(pos)
        time.sleep(LOOP_FREQUENCY)       # Need this time to move the joints


# >---> New State & Result (NEW: Add EE pose here)
        # print(f"[state] Joints: {ctrl.get_joint_states()}")  # Print new joint states
        # NEW: Get and print EE pose after action (shows effect on end-effector)
        ee_pose = ctrl.get_end_effector_pose()
        if ee_pose:
            # Round EE pose to 6 decimal places for printing
            pos_rounded = [round(x, 6) for x in ee_pose['position']]
            ori_rounded = [round(x, 6) for x in ee_pose['orientation']]
            print(f"[state] EE Pose: Pos={pos_rounded}, Ori (quaternion)= {ori_rounded}")  # Full quaternion, rounded
        else:
            print("[state] EE Pose: No data yet")

        i += 1

if __name__ == "__main__":
    main()