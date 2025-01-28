"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2025-01-27 20:24:54
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-01-28 08:10:47
FilePath: actors_nodes/attacker_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
from actors_nodes.active_node_actor import ActiveNodeActor
# This class will control the attacker node in the network

from actors_nodes.node_actor import NodeActor

class AttackerManagerActor(NodeActor):
    def __init__(self, node_id: int, attacker, num_active_attackers) -> None:
        super().__init__(node_id)
        self.node_id = node_id
        self.attacker = attacker
        self.num_active_attackers = num_active_attackers
        
    def change_behavior(self, behavior: str):
        """
        This function will change the behavior of the attackers node
        """
        message = {
            "command": "change_behavior"
        }
        
        self.broadcast(message)
        
    def on_receive(self, message: dict):
        if message.get("command") == "new_block":
            # HANDLE THE NEW BLOCK 
            self.check_scores(message)
            pass
        
        super().on_receive(message)
        
    def check_scores(self, message: dict):
        """
        This function will check the scores of the attackers
        If the cumulative score is lower than the threshold, the attacker manager will insert a new attacker in the network
        If the cumulative score is higher than the threshold, the attacker manager will remove an attacker from the network
        """
        pass
    
    def stop_attacker(self):
        """
        This function will stop one attacker
        """
        pass
        
    def add_attacker(self):
        """
        This function will add a new attacker to the network
        """
        try:
            new_attacker = ActiveNodeActor.start(len(self.partners)+1, self.attacker[self.num_active_attackers], is_attacker_flag=True)
            self.num_active_attackers += 1
            self.partners.append(new_attacker)
        except IndexError:
            print("The attacker list is empty")
            return
        
        # the idea is that the attacker will be added to the network and the verifier will tell him to start the training and in which round and the most updated model
        self.broadcast({"command": "new_attacker", "attacker": new_attacker})
        pass