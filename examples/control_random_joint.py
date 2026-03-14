import random
import time
import gz.transport as gz
from gz.msgs.double_pb2 import Double

# --------------------------
# GLOBAL JOINT LIMIT HYPERPARAMS
# --------------------------
JOINT1_MIN, JOINT1_MAX = -2.90,  2.90
JOINT2_MIN, JOINT2_MAX = -1.76,  1.76
JOINT3_MIN, JOINT3_MAX = -2.90,  2.90
JOINT4_MIN, JOINT4_MAX = -3.07, -0.07
JOINT5_MIN, JOINT5_MAX = -2.90,  2.90
JOINT6_MIN, JOINT6_MAX = -0.02,  3.75
JOINT7_MIN, JOINT7_MAX = -2.90,  2.90

FINGER1_MIN, FINGER1_MAX = 0.0, 0.04
FINGER2_MIN, FINGER2_MAX = 0.0, 0.04

# Joint ordering (including fingers)
JOINT_ORDER = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]

# Corresponding min/max values using global vars
JOINT_LIMITS = [
    (JOINT1_MIN, JOINT1_MAX),
    (JOINT2_MIN, JOINT2_MAX),
    (JOINT3_MIN, JOINT3_MAX),
    (JOINT4_MIN, JOINT4_MAX),
    (JOINT5_MIN, JOINT5_MAX),
    (JOINT6_MIN, JOINT6_MAX),
    (JOINT7_MIN, JOINT7_MAX),
    (FINGER1_MIN, FINGER1_MAX),
    (FINGER2_MIN, FINGER2_MAX),
]

def main():
    node = gz.Node()

    # Create publishers for all joints
    pubs = []
    for joint_name in JOINT_ORDER:
        topic = f"/model/panda/joint/{joint_name}/0/cmd_pos"
        pubs.append(node.advertise(topic, Double))

    print("[init] Publishers ready. Moving arm and fingers with random values...")

    while True:
        for i, pub in enumerate(pubs):
            jmin, jmax = JOINT_LIMITS[i]
            msg = Double()
            msg.data = random.uniform(jmin, jmax)
            pub.publish(msg)
        time.sleep(1)

if __name__ == "__main__":
    main()
