"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-22 15:42:13
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-12-22 15:51:41
FilePath: utils/verifier_node_actor.py
"""
import numpy as np

from actors_nodes.node_actor import NodeActor
from learning.sydelp import Sydelp
from utils.aggregation import krum
from utils.typing import FloatArray
from utils.utils import flat_weights_to_original
from keras.src.models import Model as KerasModel
from nodes.node import Node

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
    util_node: Node # used to simplify the aggregation process, it is not a real node

    # ALL THE FOLLOWING VALUES ARE PLACEHOLDERS, THEY NEED TO BE INPUTTED BY PARAMETERS
    expected_malicious_num: int = 2 # todo: revise this value
    weighting_mode: str = "uniform" # todo: revise this value
    data_sizes: list = [] # todo: revise this value
    weights_shapes: list = [] # todo: revise this value

    MIN_SCORE = 0
    
    def __init__(self,  ID, node: Node):
        super().__init__(ID)
        self.signatures = {}
        self.models = {}
        self.public_keys = {}
        self.scores = {}
        self.list_of_nodes = []
        self.round_finished = False
        self.last_block_hash = "genesis"
        self.util_node = node
        self.weights_shapes = [weight.shape for weight in self.util_node.model.get_weights()]


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

    def update_global_model(self, new_model):
        # update the global model with the new model
        try:
            self.util_node.model.set_weights(new_model)
        except Exception as e:
            print(f"Error in the update of the global model: {e}")
            return

    def create_block(self):
        # mock function to create a block of the blockchain

        # aggregate the models
        print("Aggregating the models")
        new_model = self.aggregation()

        # update the global model
        self.update_global_model(new_model)

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
            "model": new_model,
            "scores": {},
            "round": self.round,
            "hash": "MockHash",
            "last_block_hash": self.last_block_hash
        }
        print(f"Block {self.round} created")

        return message

    def update_data_sizes(self):
        # update the data sizes
        # the data sizes are the number of rows of the data in the nodes
        models_length = []
        for id_node in self.models.keys():
            models_length.append(len(self.models[id_node]))
        self.data_sizes = np.array(models_length)


    def aggregation(self):
        # updated version of the aggregation function in the sydelp class

        # step 1: update model matrix
        # list of models is a list of keras models
        # get the list of models from the dictionary
        # print("Getting the list of models")
        # print("Models keys: ", self.models.keys())

        list_of_models = [self.models[id_node] for id_node in self.models.keys()]
        # print("List of models: ", list_of_models)
        self.update_data_sizes()

        # create the matrix of models
        # print("Creating the models matrix")
        try:
            models_matrix = np.empty((len(self.models),
                                      len(self.models[3])),
                                     dtype="float")
        except Exception as e:
            print(f"Error in the creation of the models matrix: {e}")
            return

        for i, model in enumerate(list_of_models):
            models_matrix[i] = model
        # matrix_of_models = [model.flatten() for model in self.models]
        # print("Models matrix: ", models_matrix)

        # step 2: get krums result and labels (+1, -1)
        # the result is the weights of the model
        # TODO: modify the krum function to return the weights of the model and the labels [(id_node, label)]
        try:
            print("Averaging with krum")
            krum_result =  krum(models_matrix,
                 m=self.expected_malicious_num,
                 weighting_mode=self.weighting_mode,
                 data_sizes=self.data_sizes)
        except Exception as e:
            print(f"Error in the krum function: {e}")
            return

        # print("Krum results: ", krum_result)

        # print("Weights shapes: ", self.weights_shapes)

        # step 3: get the weights of the model
        avg_model = flat_weights_to_original(krum_result, self.weights_shapes)
        # print("Aggregated model: ", avg_model)


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

        # reset the models
        self.models = {}

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
            print("Round finished: ", self.round)
            print("Models length: ", len(self.models.keys()))
            print("Partners length: ", len(self.partners))
            self.end_round()
