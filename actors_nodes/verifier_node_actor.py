"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-22 15:42:13
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-12-22 15:51:41
FilePath: utils/verifier_node_actor.py
"""
from actors_nodes.node_actor import NodeActor


class VerifierNodeActor(NodeActor):
    list_of_nodes: list
    list_of_public_keys: list
    list_of_signatures: list
    list_of_scores: list
    round_finished: bool
    round: int
    
    def __init__(self,  ID):
        super().__init__( ID)
        self.list_of_nodes = []
        self.list_of_public_keys = []
        self.list_of_signatures = []
        self.list_of_scores = []
        self.round_finished = False

    def verify(self, public_key, signature):
        # mock function to verify the signature and the PoW
        return True
    
    def create_block(self):
        # mock function to create a block of the blockchain
        pass
    
    def aggregate(self):
        # use the aggregated functions in aggregation.py
        pass
    
    def add_model(self):
        # add the model to the list of models
        pass
    
    def calculate_new_scores(self):
        # calculate the new scores
        pass
    
    def end_round(self):
        # end the round
        
        # aggregate the models

        # calculate the new scores
        
        # update the scores in the nodes

        # create the block

        # broadcast the block

        # reset the round

        pass
    
