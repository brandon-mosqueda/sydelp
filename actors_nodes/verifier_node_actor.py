"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-22 15:42:13
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-12-22 15:51:41
FilePath: utils/verifier_node_actor.py
"""
from actors_nodes.node_actor import NodeActor
from learning.sydelp import Sydelp
from utils.aggregation import krum
from utils.typing import FloatArray
from utils.utils import flat_weights_to_original
from keras.src.models import Model as KerasModel


class VerifierNodeActor(NodeActor):
    """
    Verifier Node Actor class

    This class is used to implement the verifier node actor in the federated learning process.

    The verifier node is responsible for aggregating the models from the trainers and for verifying the signatures, and verifying the PoW.

    The verifier node is also responsible for creating the blocks of the blockchain.
    """

    number_of_nodes: list
    public_keys: dict # {node_id: public_key}
    signatures: dict # {node_id: signature}
    scores: dict # {node_id: score}
    models: dict # {node_id: model}
    round_finished: bool
    last_block_hash: str

    # ALL THE FOLLOWING VALUES ARE PLACEHOLDERS, THEY NEED TO BE INPUTTED BY PARAMETERS
    expected_malicious_num: int = 2 # todo: revise this value
    weighting_mode: str = "uniform" # todo: revise this value
    data_sizes: list = [] # todo: revise this value
    weights_shapes: list = [] # todo: revise this value

    MIN_SCORE = 0
    
    def __init__(self,  ID):
        super().__init__(ID)
        self.signatures = {}
        self.models = {}
        self.public_keys = {}
        self.scores = {}
        self.list_of_nodes = []
        self.round_finished = False


    def on_receive(self, message):
        # print(f"Node {self.ID} received a message: {message['command']}")

        if message.get('command') == 'new_model':
            # store the information from the trainers
            print(f"Node {self.ID} received a new model")
            try:
                self.store_model(message)
            except KeyError:
                print("The model is not present in the message")

            return

        # THE ADDING OF A NEW NODE IS HANDLED BY THE VERIFIER
        if message.get('command') == 'new_node':
            # a new node is participating in the training we need to add it to the list of nodes
            self.add_node(message)

            # store the public key
            self.store_public_key(message)

            # add the maximum score to the list of scores
            self.list_of_scores[message['ID']] = VerifierNodeActor.MIN_SCORE

            # send the message to the new node to start the training
            self.send_start_training_new_node(message)

            return

        if message.get('command') == 'DEBUG':
            self.print_list_of_nodes()
            return

        # handle the other messages
        super().on_receive(message)

    def add_node(self, message):
        # add the node to the list of nodes and the list of partners

        self.list_of_nodes.append(message['node_id']) # add the node to the list of nodes (the ID of the node)

        new_partners = self.partners + [message['node']]

        self.update_partners(new_partners)


    def print_list_of_nodes(self):
        print(self.list_of_nodes)

    def store_public_key(self, message):
        # store the public key of the new node
        self.list_of_public_keys.append(message['public_key'])

    def verify(self, public_key, signature):
        # mock function to verify the signature and the PoW
        return True
    
    def create_block(self):
        # mock function to create a block of the blockchain

        # aggregate the models
        # self.aggregation()

        # calculate the new scores and update the scores in the nodes
        # TODO: implement the calculation of the new scores

        # create Hash
        # TODO: implement the creation of the hash

        # add last block Hash
        if self.round == 0:
            self.last_block_hash = "genesis"
        else:
            self.last_block_hash = "MockHash"

        # create the block
        message = {
            "command": "new_block",
            "model": "TODO",
            "scores": {},
            "round": self.round,
            "hash": "MockHash",
            "last_block_hash": self.last_block_hash
        }

        return message


    def aggregation(self):
        # updated version of the aggregation function in the sydelp class

        # step 1: update model matrix
        # list of models is a list of keras models
        # get the list of models from the dictionary
        list_of_models = [self.models[id_node] for id_node in self.models.keys()]
        matrix_of_models = [model.flatten() for model in self.models]

        # step 2: get krums result and labels (+1, -1)
        # the result is the weights of the model
        # TODO: modify the krum function to return the weights of the model and the labels [(id_node, label)]
        krum_result =  krum(self.models_matrix,
                 m=self.expected_malicious_num,
                 weighting_mode=self.weighting_mode,
                 data_sizes=self.data_sizes)

        print("Krum results: ", krum_result)

        # step 3: get the weights of the model
        avg_model = flat_weights_to_original(krum_result, self.weights_shapes)


        # we just need to exchange the model weights since the trainers nodes can update the model easily
        return avg_model



    def add_model(self):
        # add the model to the list of models
        pass
    
    def calculate_new_scores(self):
        # calculate the new scores
        pass
    
    def end_round(self):
        # end the round

        # create the block
        new_block = self.create_block()

        # broadcast the block
        self.broadcast(new_block)

        # update the round
        self.round += 1

        # # send the message to the trainers to start the new round
        self.broadcast({"command": "start", "round": self.round})


    
    def send_start_training_new_node(self, message):
        # send the start training message to the new node
        new_message = {
            "command": "start",
            "round": self.round
        }

        message['node'].tell(new_message)


    def store_model(self, message):
        # store the model from the trainers
        try:
            self.models[message['ID']] = message['model']
        except KeyError:
            print("The model is not present in the message")
            return

        # store the signature
        try:
            self.signatures[message['ID']] = message['signature']
        except KeyError:
            print("The signature is not present in the message")
            return

        # verify the signature TODO: implement the verification
        if not self.verify(0, 0):
            print("The signature is not valid")
            return

        # check if the round is finished
        if len(self.models.keys()) >= len(self.partners) - 1 :
            self.end_round()
