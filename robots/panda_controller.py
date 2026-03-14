# connect_0_2.py
"""
This file provides options to control the robot.
Main Features:
set_positions()
reset()
start_joint_monitoring()            # set up a subscriber to be able to get joint states
get_joint_states()                  # Return all joint states: {'panda_joint1': 2.27,....'panda_finger_joint2': -1.45}
"""
import gz.transport as gz
from gz.msgs.double_pb2 import Double
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.model_pb2 import Model
from gz.msgs import model_pb2
import time

JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
LIMITS = [(-2.90, 2.90), (-1.76, 1.76), (-2.90, 2.90), (-3.07, -0.07), (-2.90, 2.90), (-0.02, 3.75), (-2.90, 2.90), (0, 0.04), (0, 0.04)]

class PandaController:
    def __init__(self, joint_names, world="panda_world"):
        self.node = gz.Node()
        self.pubs = {j: self.node.advertise(f"/model/panda/joint/{j}/0/cmd_pos", Double) for j in joint_names}
        self.control_svc = f"/world/{world}/control"
        self.joint_names = joint_names
        self.latest_joint_msg = None
        self.start_joint_monitoring()

    def set_positions(self, positions):
        for j, val in positions.items():
            msg = Double(); msg.data = val
            self.pubs[j].publish(msg)

    def reset(self):
        req = WorldControl(); req.reset.all = True
        ok, rep = self.node.request(self.control_svc, req, WorldControl, Boolean, 3000)
        print("[reset] success" if ok and rep.data else "[reset] failed")
        time.sleep(0.1)  # Small delay for stability

    def start_joint_monitoring(self):   # set up a subscriber to the topic: "/model/panda/joint_state", called in constructor
        def joint_callback(raw_msg, _info):
            nonlocal self
            parsed = model_pb2.Model()
            parsed.ParseFromString(raw_msg)
            self.latest_joint_msg = parsed
        msg_type_str = "gz.msgs.Model"
        options = gz.SubscribeOptions()
        self.node.subscribe_raw("/model/panda/joint_state", joint_callback, msg_type_str, options)
        # print("[init] Listening for joint states...")

    def get_joint_states(self):         # Returns a dictionary of joint states, won't function without start_joint_monitoring()
        joint_states = {j: None for j in self.joint_names}
        if not self.latest_joint_msg or not self.latest_joint_msg.joint:
            print("[get_joint_states] No joint state data received")
            return joint_states
        for joint in self.latest_joint_msg.joint:
            if joint.name in joint_states and joint.axis1 and joint.axis1.position is not None:
                joint_states[joint.name] = joint.axis1.position
            elif joint.name in joint_states:
                print(f"[get_joint_states] No position data for joint {joint.name}")
        return joint_states