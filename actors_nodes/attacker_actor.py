"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2025-01-27 20:24:54
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-02-04 17:04:21
FilePath: actors_nodes/attacker_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
from random import random

import numpy as np

from actors_nodes.active_node_actor import ActiveNodeActor
# This class will control the attacker node in the network

from actors_nodes.node_actor import NodeActor

class AttackerManagerActor(NodeActor):
    def __init__(self, node_id: int, attacker, num_active_attackers, offset_ids, alpha, max_computing_power, T) -> None:
        super().__init__(node_id)
        self.attacker = attacker
        self.num_active_attackers = num_active_attackers
        self.new_attacker_id = num_active_attackers + offset_ids
        self.offset_ids = offset_ids
        self.alpha = alpha  
        self.round = 0
        self.COMPUTING_POWER = max_computing_power
        self.T = T
        self.debug()
        self.attack_started_flag = False
    
    def debug(self):
        print(f"Attacker Manager Actor {self.ID}")
        print(f"Attacker: {self.attacker}")
        print(f"Number of active attackers: {self.num_active_attackers}")
        
    def change_behavior(self, behavior: str):
        """
        This function will change the behavior of the attackers node
        """
        message = {
            "command": "change_behavior"
        }
        
        self.broadcast(message)
        
    def check_change_behavior(self):
        if self.num_active_attackers > int(((self.offset_ids -4)+self.num_active_attackers)* 0.4) :
            self.change_behavior("attack")
            self.attack_started_flag = True
        
    def on_receive(self, message: dict):
        # print(f"Attacker Manager Actor {self.ID} received a message: {message.get('command')}")
        if message.get("command") == "new_block":
            # HANDLE THE NEW BLOCK 
            print(f"Attacker Manager Actor {self.ID} received a new block")
            self.check_scores(message)
            
            print(f"Attacker Manager Actor {self.ID} is checking the change of behavior")
            if not self.attack_started_flag:
                self.check_change_behavior()
            
            return
        
        super().on_receive(message)
        
    def compute_difficulty(self, score: float):
        """
        This function will compute the difficulty of the PoW in range [0, 1]
        """
        if score == 0:
            return 1.
        
        tmp = (self.T - self.alpha)/(self.T - 1)
        
        res = tmp ** score
        
        if res > 1:
            res = 1
            
        return res
        
    def check_scores(self, message: dict):
        """
        This function will check the scores of the attackers
        If the cumulative score is lower than the threshold, the attacker manager will insert a new attacker in the network
        If the cumulative score is higher than the threshold, the attacker manager will remove an attacker from the network
        """
        self.round = message.get("round")
        
        sum_score = 0
        scores = message.get("scores")
        
        print(f"Scores (attacker): {scores}")
        attackers_id = [i for i in range(self.offset_ids, self.num_active_attackers+self.offset_ids)]
        print(f"Attackers ID: {attackers_id}")
        
        
        for i in attackers_id:
            sum_score += self.compute_difficulty(scores[i])
            
        print(f"Sum of the scores attackers: {sum_score}")

        # take the sum of the scores of some random honest nodes
        random_honest_nodes = []
        choichable_nodes = [i for i in range(4, self.offset_ids)]
        for i in range(len(attackers_id)):
            # pick a random honest node in choichable_nodes
            node = np.random.choice(choichable_nodes)
            random_honest_nodes.append(node)
            choichable_nodes.remove(node)
            
        sum_score_rand = 0
        for i in random_honest_nodes:
            sum_score_rand += self.compute_difficulty(scores[i])
        
        print(f"Sum of the scores honest nodes: {sum_score_rand}")
        
        message = {
            "command": "scores",
            "round": self.round,
            "attackers_scores": sum_score,
            "honest_scores": sum_score_rand
        }
        
        self.broadcast(message)
        
        diff = self.COMPUTING_POWER - sum_score
        
        while diff > 1:
            # add an attacker
            print("Adding an attacker")
            self.add_attacker()
            diff -= 1
            
        if diff < 0:
            # THIS SHOULDN'T HAPPEN
            print("Error: too many attackers in the network")
        return
    
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
            print("Trying to add a new attacker to the network")
            try:
                new_attacker = ActiveNodeActor.start(self.new_attacker_id, self.attacker[self.num_active_attackers], is_attacker_flag=True)
            except Exception as e:
                print(f"Error in the creation of the new attacker: {e}")
            self.num_active_attackers += 1
            self.partners.append(new_attacker)
            print("New attacker added to the network: ", new_attacker)
            new_attacker.tell({"command": "set_partners", "partners": self.partners[:2]})
        except IndexError:
            print("The attacker list is empty")
            return
        
        # the idea is that the attacker will be added to the network and the verifier will tell him to start the training and in which round and the most updated model
        self.broadcast({"command": "new_attacker", "node": new_attacker, "node_id": self.new_attacker_id})
        self.new_attacker_id += 1
        pass