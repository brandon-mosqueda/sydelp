"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-28 12:42:14
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-01-06 10:54:01
FilePath: actors_nodes/active_node_actor.py
"""
from math import ceil

# actor that participate in the training process, both honest and malicious trainers
from actors_nodes.node_actor import NodeActor
from nodes.node import Node

class ActiveNodeActor(NodeActor):
    """
    Active Node Actor class

    *Description*

    This class is used to implement the active node actor in the federated learning process. (both honest and malicious)

    The active node is responsible for training the model and for attacking the system.

    The main function of the training process is directly retireved from the "./nodes" package, so we don't need to reimplement it.

    The functionality are transposed by using the node attribute of the class as a reference to the node object in the nodes package.
    """

    node: Node # this node refers to the node object in the nodes package, the implementation is already done,
    # we will exploit the built-in functions, so we don't need to reimplement the functions for attacking and training
    public_key: str
    private_key: str
    secret_key: str
    score: float # score to compute the difficulty of the PoW, range  [0, 1]
    first_round: bool # flag to check if it is the first round of the training
    is_attacker: bool # flag to check if the node is an attacker
    
    def __init__(self,  node_id: int, node: Node, is_attacker_flag: bool) -> None:
        super().__init__(node_id)
        self.node = node
        self.public_key = "public_key_placeholder"
        self.secret_key = "secret_key_placeholder"
        self.score = 0. # the score is the difficulty of the PoW, maximum score means maximum difficulty
        self.first_round = True
        self.is_attacker = is_attacker_flag

    def on_receive(self, message: dict):        
        # 2 handle the starting message, it will start the training process
        if message.get('command') == 'start':
            if self.first_round:
                self.first_round = False
                try:
                    self.round = message['round']
                    # if first round the model is not present in the message so we will use the init model
                except KeyError:
                    print("\nThe round is not present in the message, it is the first round\n")
                    self.round = 0

            print(f"training started for node {self.ID} in round {self.round}")

            # # TODO: check if the same round (in case of more then one verifier
            # else:
            #     try:
            #         # TODO: check the index of the ID
            #         self.update_score(message['scores'][self.ID])
            #     except KeyError:
            #         print("The score is not present in the message")
            #         return
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
        # print(f"Node {self.ID} received a new block: {block}")

        # corner case there are two or more verifiers, so check if the block is already processed by checking the round
        if block['round'] <= self.round:
            return

        # update the round, should be the same for all the nodes
        self.round = block['round']

        # update the score
        # self.update_score(block['scores'][self.ID])

        # update the model, check this function
        self.node.set_model_weights(block['model'])


    # FOR NOW WE WILL NOT IMPLEMENT THE FOLLOWING FUNCTIONS, WE WILL IMPLEMENT THEM LATER
    def add_new_node(self, node):
        # The idea is that the attacker when reaching a good score can add a new attacker node to the training process
        # TODO: implement  the creation of a new node

        new_node = NodeActor.start(self.partners + 2, Node(), len(self.partners))
        
        self.partners.append(new_node)
        
        # REVISE THIS
        
        self.broadcast({"command": "new_node", "node": new_node, "public_key": new_node.public_key, "node_id": len(self.partners)})
        
        
    def start_training(self):
        # do the new training, with the built-in node object
        self.node.train()

        if self.is_attacker:
            # if the node is an attacker, generate a new model
            try:
                # TODO: we will change the if in to a more dynamic choice of the attack
                self.node.attack()
            except Exception as e:
                print(f"Error in the attack: {e}")
                return

        # calculate the proof of work
        (y, pi) = self.calculate_VDP()

        # generate the signature
        signature = self.generate_signature()

        # broadcast our new model
        # TODO: this is just the structure of the message, we need to implement the actual functions, like signature generation, proof of work, etc.
        msg = {
            "command": "new_model",
            "model": self.node.get_flat_model_weights(), # dont know if this is the correct function CHECK IT
            "signature": signature,
            "proof_of_work": (y, pi),
            "public_key": self.public_key,
            "ID": self.ID,
            "round": self.round
        }

        self.broadcast(msg)
        
    def calculate_VDP(self):
        # calculate the Verifiable Delay Puzzle
        # (y, pi) = VDP(public_key, signature, H  hash reference to the last block, Dc difficulty of the challenge)
        # y is the result, pi is the proof
        # calculate the proof of work with: the public key, the signature,
        return ("Mock y", "Mock pi")

    def calculate_Dc(self):
        # calculate the difficulty of the challenge
        D = 100 # maximum difficulty (initial difficulty)
        T = self.round
        alpha = 1.5 # this is an hyperparameter to be chosen by the experimenter
        f_score = ((T-alpha)/(T-1))**self.score

        return ceil(D*f_score)
    
    def generate_signature(self):
        # generate the signature with: the model, the secret key, the public key
        return "Mock signature"

    def update_score(self, score):
        # # update the score
        # self.score = score
        return