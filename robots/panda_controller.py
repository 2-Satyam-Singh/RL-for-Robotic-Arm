"""
This file provides options to control the robot.
Main Features:
set_positions()
reset()
"""
import gz.transport as gz
from gz.msgs.double_pb2 import Double
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean

class PandaController:
    def __init__(self, joint_names, world="panda_world"):
        self.node = gz.Node()
        self.pubs = {
            j: self.node.advertise(f"/model/panda/joint/{j}/0/cmd_pos", Double)
            for j in joint_names
        }
        self.control_svc = f"/world/{world}/control"

    def set_positions(self, positions):
        for j, val in positions.items():
            msg = Double(); msg.data = val
            self.pubs[j].publish(msg)

    def reset(self):
        req = WorldControl(); req.reset.all = True
        ok, rep = self.node.request(self.control_svc, req, WorldControl, Boolean, 3000)
        print("[reset] success" if ok and rep.data else "[reset] failed")
