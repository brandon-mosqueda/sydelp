"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-22 15:42:13
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-12-22 15:51:41
FilePath: utils/verifier_node_actor.py
"""
from math import floor

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
    
    def __init__(self,  ID, node: Node, T, exp_num_malicious) -> None:
        super().__init__(ID)
        self.signatures = {}
        self.models = {}
        self.public_keys = {}
        self.scores = {}
        self.list_of_nodes = [i for i in range(3, len(self.partners) + 3)]
        self.round_finished = False
        self.last_block_hash = "genesis"
        self.util_node = node
        self.weights_shapes = [weight.shape for weight in self.util_node.model.get_weights()]
        self.FirstRound = True
        self.expected_malicious_num = exp_num_malicious
        self.T = T

    def init_scores(self):
        # initialize the scores
        # print("sorted models keys: ", sorted(self.models.keys()))
        for node_id in self.models.keys():
            self.scores[node_id] = VerifierNodeActor.MIN_SCORE

    def on_stop(self):
        self.broadcast({"command": "stop"})
        self.stop()

    def on_receive(self, message):
        # print(f"Node {self.ID} received a message: {message['command']}")

        if message.get('command') == 'new_model':
            # store the information from the trainers
            # print(f"Node {self.ID} received a new model")
            try:
                self.store_model(message)
            except KeyError:
                print("The model is not present in the message")

            return

        # THE ADDING OF A NEW NODE IS HANDLED BY THE VERIFIER
        if message.get('command') == 'new_attacker':
            # a new node is participating in the training we need to add it to the list of nodes
            self.add_node(message)
            return

        if message.get('command') == 'DEBUG':
            self.print_list_of_nodes()
            return

        # handle the other messages
        super().on_receive(message)

    def add_node(self, message):
        # add the node to the list of nodes and the list of partners
       #  print("Adding a new node: ", message)

        self.list_of_nodes.append(message.get('node_id')) # add the node to the list of nodes (the ID of the node)

        new_partners = self.partners + [message.get('node')] # add the node to the list of partners

        self.update_partners(new_partners)

        # print("New node added: ", message.get('node'))

        self.scores[message.get('node_id')] = 0

        # print("Added the new score to the new node")

        # send the last global model to the new node
        new_message = {
            "command": "init_node",
            "model": self.util_node.get_model_weights(),
            "round": self.round
        }

        message['node'].tell(new_message)
        # print("Message init sent to the new node")

        # send the start training message to the new node
        start_message = {
            "command": "start",
            "round": self.round
        }

        message['node'].tell(start_message)
        # print("Message start sent to the new node")


    def print_list_of_nodes(self):
        print(self.list_of_nodes)

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
        # print("Aggregating the models")
        if self.FirstRound:
            self.init_scores()
            self.FirstRound = False
        # print("scores before aggregation: ", self.scores)
        # print("len of models: ", len(self.models))
        # print("models keys: ", self.models.keys())
        new_model = self.aggregation()
        # print("scores after aggregation: ", self.scores)

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
            "scores": self.scores,
            "round": self.round,
            "hash": "MockHash",
            "last_block_hash": self.last_block_hash
        }
        # print(f"\nBlock {self.round} created\n")

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
        sorted_keys = sorted(self.models.keys())
        # print("Sorted keys: ", sorted_keys)
        list_of_models = [self.models[id_node] for id_node in sorted_keys]
        self.update_data_sizes()

        try:
            models_matrix = np.empty((len(self.models),
                                      len(self.models[4])),
                                     dtype="float")
        except Exception as e:
            print(f"Error in the creation of the models matrix: {e}")
            return

        for i, model in enumerate(list_of_models):
            models_matrix[i] = model

        # TODO: modify the krum function to return the weights of the model and the labels [(id_node, label)]
        expected_malicious_num = int(round((len(self.models) * self.expected_malicious_num),0))
        # print("Expected malicious number: ", expected_malicious_num)
        try:
            krum_result =  krum(models_matrix,
                 m= expected_malicious_num,
                 weighting_mode=self.weighting_mode,
                 data_sizes=self.data_sizes)
        except Exception as e:
            print(f"Error in the krum function: {e}")
            return

        # print("Best indexes: ", krum_result[1])
        self.update_scores(krum_result[1])

        # step 3: get the weights of the model
        avg_model = flat_weights_to_original(krum_result[0], self.weights_shapes)
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
        if self.round < self.T:
            self.broadcast({"command": "start", "round": self.round})
        else:
            self.stop()


    
    def send_start_training_new_node(self, message):
        # send the start training message to the new node
        new_message = {
            "command": "start",
            "round": self.round
        }

        message['node'].tell(new_message)
        # print("Message start sent to the new node")

    def update_scores(self, krum_indexes):
        # print("Length of krum indexes: ", len(krum_indexes))
        tmp_flag_map = [0] * (len(self.partners) - 2)

        for i in range(len(krum_indexes)):
            node_id = krum_indexes[i] + 4 # check this
            if node_id in self.scores.keys():
                self.scores[node_id] += 1
            else:
                self.scores[node_id] = 1
            tmp_flag_map[krum_indexes[i]] = 1

        for i in range(len(tmp_flag_map)):
            if tmp_flag_map[i] == 0:
                if i + 4 in self.scores.keys():
                    if self.scores[i + 4] > 0:
                        self.scores[i + 4] -= 1

                else:
                    self.scores[i + 4] = 0


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
        if len(self.models.keys()) >= len(self.partners) - 2:
            self.end_round()
