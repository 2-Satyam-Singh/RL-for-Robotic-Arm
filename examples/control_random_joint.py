# control_0_4.py
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
LOOP_FREQUENCY = 1

def main():
    ctrl = PandaController(JOINTS)
    print("[init] Panda controller active.")
    ctrl.start_pose_monitoring()

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

        # pos = ctrl.get_entity_positions()  # Print latest entity positions
        # print(f"[state] Positions: {pos}\n")


# >---> Action Performed
        pos = {j: random.uniform(*lim) for j, lim in zip(JOINTS, LIMITS)}  # Random actions
        # OR
        # pos = {j: js[j] + 1 for j, lim in zip(JOINTS, LIMITS)}  # This can sometimes fail, because sometimes the value for all keys returned by js is "None", but at other times, it gives proper values, so works fine... it's just random.. need a way to handle in case of 'None'

        ctrl.set_joint_positions(pos)
        time.sleep(LOOP_FREQUENCY)       # Need this time to move the joints


# >---> New State & Result
        # print(f"[state] Joints: {ctrl.get_joint_states()}")  # Print new joint states


        i += 1

if __name__ == "__main__":
    main()
