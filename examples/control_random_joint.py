"""
This is the main file and it currently perform 
JOINT VALUE UPDATIONS
RESET the environment
"""

#   Run as: python3 -m examples.control_random_joint

import random, time
from robots.panda_controller import PandaController

JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1","panda_finger_joint2"]
LIMITS = [(-2.9,2.9),(-1.76,1.76),(-2.9,2.9),(-3.07,-0.07),(-2.9,2.9),(-0.02,3.75),(-2.9,2.9),(0,0.04),(0,0.04)]

def main():
    ctrl = PandaController(JOINTS)
    print("[init] Panda controller active.")
    i = 0
    while True:
        pos = {j: random.uniform(*lim) for j, lim in zip(JOINTS,LIMITS)}
        ctrl.set_positions(pos)
        time.sleep(1)
        i += 1
        if i % 10 == 0:
            print("[reset] Resetting the world.")
            ctrl.reset()

if __name__=="__main__": main()
