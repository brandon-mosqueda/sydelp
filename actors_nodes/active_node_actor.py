"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-28 12:42:14
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-01-02 12:53:03
FilePath: actors_nodes/active_node_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
# actor that participate in the training process, both honest and malicious trainers
from actors_nodes.node_actor import NodeActor
from nodes.node import Node

class ActiveNodeActor(NodeActor):
    node: Node # this node refers to the node object in the nodes package, the implementation is already done,
    # we will exploit the built-in functions, so we don't need to reimplement the functions for attacking and training
    public_key: str
    private_key: str
    score: float # score to compute the difficulty of the PoW, range  [0, 1]
    first_round: bool # flag to check if it is the first round of the training
    
    def __init__(self, node: Node, node_id: int) -> None:
        super().__init__( node_id)
        self.node = node
        self.public_key = "public_key_placeholder"
        self.score = 0 # the score is the difficulty of the PoW, maximum score means maximum difficulty
        self.first_round = True

    def on_receive(self, message: dict):        
        # 2 handle the starting message, it will start the training process
        if message.get('command') == 'start':
            if self.first_round:
                self.first_round = False
                try:
                    self.round = message['round']
                except KeyError:
                    print("The round is not present in the message")
                    self.round = 0
            else:
                try:
                    # TODO: check the index of the ID
                    self.update_score(message['scores'][self.ID])
                except KeyError:
                    print("The score is not present in the message")
                    return
            self.start_training()
            return
        
        # 3 handle the new block message
        if message.get('command') == 'new_block':
            self.read_new_block(message)
            return
        
        # 4 handle the new model message
        if message.get('command') == 'new_model':
            return
        
        # 5 all the other messages are handled by the parent class
        super().on_receive(message)
    

            
    # only the active nodes should read the new block, in fact for the verifiers the new block is already processed and the UI should not read the new block
    def read_new_block(self, block):
        """Read the new block from the blockchain."""


        # corner case there are two or more verifiers, so check if the block is already processed by checking the round
        if block['round'] == self.round:
            return

        # update the round
        self.round += 1
        
    def start_training(self):
        # do the new training, with the built-in node object
        # TODO: with the built-in node object

        # calculate the proof of work
        PoW = self.calculate_PoW()

        # generate the signature
        signature = self.generate_signature()

        # broadcast our new model
        # TODO: this is just the structure of the message, we need to implement the actual functions, like signature generation, proof of work, etc.
        msg = {
            "command": "new_model",
            "model": self.node.get_model_weights(), # dont know if this is the correct function CHECK IT
            "signature": signature,
            "proof_of_work": PoW,
            "public_key": self.public_key,
            "ID": self.ID,
            "round": self.round
        }

        self.broadcast(msg)
        
    def calculate_PoW(self):
        # calculate the proof of work
        return "Mock PoW"
    
    def generate_signature(self):
        # generate the signature
        return "Mock signature"