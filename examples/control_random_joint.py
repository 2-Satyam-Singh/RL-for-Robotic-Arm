"""
This is the main file and it currently perform 
JOINT VALUE UPDATIONS
RESET the environment
RECIVENG JOINT STATE
"""
import random, time
from robots.panda_controller import PandaController, JOINTS, LIMITS

# ──────── Removed finger joints for training (only 7 arm joints now) ────────
ARM_JOINTS = JOINTS[:-2]      # panda_joint1..7 only
ARM_LIMITS = LIMITS[:-2]

# Hyperparameter — will be loaded from .yaml later
DECIMAL_PLACES = 1   # ← change here (or in yaml) to 2 / 0 / 0.5 etc.

def main():
    ctrl = PandaController(JOINTS)
    print(f"[init] Panda controller active (fingers excluded, precision = {DECIMAL_PLACES} decimal places).")
    i = 0
    while True:
# >---> Reset
        # if i % 100 == 0:
        #     print("[reset] Resetting the world.")
        #     ctrl.reset()

# >---> Previous State
        js = ctrl.get_joint_states()  # Get joint states
        js = {j: round(v, DECIMAL_PLACES) if v is not None else None for j, v in js.items()}  # restricted precision
        print(f"[state] Joints: {js}")  # Print old joint states

# >---> Action Performed
        pos = {j: round(random.uniform(*lim), DECIMAL_PLACES) for j, lim in zip(ARM_JOINTS, ARM_LIMITS)} # Random actions (restricted)
        # OR
        # pos = {j: round(js[j] + 1, DECIMAL_PLACES) for j, lim in zip(ARM_JOINTS, ARM_LIMITS)} # now safe, no None issue
        
        ctrl.set_positions(pos)
        time.sleep(1)

# >---> New State & Result
        js = ctrl.get_joint_states()
        js = {j: round(v, DECIMAL_PLACES) if v is not None else None for j, v in js.items()}  # restricted precision
        print(f"[state] Joints: {js}") # Print new joint states (now rounded)

        i += 1


if __name__ == "__main__":
    main()