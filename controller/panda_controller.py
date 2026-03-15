# connect_0_3.py
"""
These functions are available:
Main Features:
set_joint_positions()
reset()
start_joint_monitoring()            # toset up a subscriber to be able to get joint states
get_joint_states()                  # Return all joint states: {'panda_joint1': 2.27,....'panda_finger_joint2': -1.45}
start_pose_monitoring()
get_entity_positions()
"""
import gz.transport as gz
from gz.msgs.double_pb2 import Double
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs import model_pb2, pose_pb2
import gz.msgs.pose_v_pb2 as pose_v_pb2
import time

class PandaController:
    ENTITIES = {
        "Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure",
        "Avengers_Thor_PLlrpYniaeB",
        "My_Little_Pony_Princess_Celestia"
    }

    def __init__(self, joint_names, world="panda_world"):
        self.node = gz.Node()
        self.pubs = {j: self.node.advertise(f"/model/panda/joint/{j}/0/cmd_pos", Double) for j in joint_names}
        self.control_svc = f"/world/{world}/control"
        self.joint_names = joint_names
        self.latest_pose_msg = None
        self.latest_joint_msg = None
        self.start_joint_monitoring()

    def set_joint_positions(self, positions):
        for j, val in positions.items():
            msg = Double(); msg.data = val
            self.pubs[j].publish(msg)

    def reset(self):
        req = WorldControl(); req.reset.all = True
        ok, rep = self.node.request(self.control_svc, req, WorldControl, Boolean, 3000)
        print(f"[reset] success, {rep.data}" if ok and rep.data else f"[reset] failed {rep.data}")
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

    def start_pose_monitoring(self):
        def pose_callback(raw_msg, _info):
            msg = pose_v_pb2.Pose_V()
            msg.ParseFromString(raw_msg)
            self.latest_pose_msg = msg
        self.node.subscribe_raw(
            "/world/panda_world/pose/info",
            pose_callback,
            "gz.msgs.Pose_V",
            gz.SubscribeOptions()
        )
        print("[init] Listening for entity positions...")

    def get_entity_positions(self):             # BUG There is a bug here, sometimes returns non-int value "None" other times even after reset it keeps old positions, let's NOT fix the bug, because it might be a server side bug and we don't need it either, we want to give it random positions anyway independent of current positions
        entities = {}
        if not self.latest_pose_msg:
            print("[get_entity_positions] No pose data received yet")
            return entities
        for pose in self.latest_pose_msg.pose:
            if pose.name in self.ENTITIES:
                p = pose.position
                entities[pose.name] = [p.x, p.y, p.z]
        return entities