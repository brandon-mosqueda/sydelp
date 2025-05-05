from __future__ import annotations

from actors_nodes.active_node_actor import ActiveNodeActor
from actors_nodes.attacker_actor import AttackerManagerActor
from actors_nodes.user_interface_actor import UserInterfaceActor
from actors_nodes.data_logger_actor import DataLoggerActor
from actors_nodes.verifier_node_actor import VerifierNodeActor
from nodes_builder import nodes_builder
from nodes.node import Node
import utils.initialize as init
import utils.utils as utils

from typing import Union
from nodes.node import Node
from utils.utils import IntArray, as_name, NumArray
from attack.attacker import Attacker
from learning.learning import Learning
from learning.learning import MetricParams
from keras.src.models import Model as KerasModel
import numpy as np
import json

OFFSET = 4


def read_input_params(input_file_path: str) -> dict:
    # read the input parameters from ./params_network/1_input_params.json
    try:
        # here specify the path to the input parameters file
        with open(input_file_path, "r") as f:
            input_params = json.load(f)
    except Exception as e:
        print(f"Error in reading the input parameters: {e}")
        return None

    return input_params


DEBUG = False


def run_network_implementation(input_file_path: str):
    """
    This function is the main entry point for the network implementation.
    :param input_file_path: file to read the input parameters
    :return: None (it prints the results in the output file)
    """

    print("Starting the MultiAgent network implementation of Sydelp Algorthm...\n")
    print(f"Input Parameters file:{input_file_path} \n\n")

    # this list will contain all the actor, the actor are the agents participating (not all actively) in the network.
    # The agents (or actors) actually participating in the training process (trainers, attackers and verifier)
    # will have associated a node object, with the meaning of the data structure "../nodes/node.py".
    # These nodes object allow the computation of the already built functions, (mainly train and attack)
    # also manages the type of attack and the datasets
    # for the purpose of this part of the implementation all these latter functions parameters are to be considered as hyperparameters
    actors = []

    if DEBUG:
        print(f"Starting the network, with mock structure")

    try:
        input_params = read_input_params(input_file_path)
        HONEST_NODES = input_params["HONEST_NODES"]
        INIT_MALICIOUS_NODES = input_params["INIT_MALICIOUS_NODES"]
        ALPHA = float(input_params["alpha"])
        exp_num_of_attacks = float(input_params["exp_num_of_attackers"])
        T = float(input_params["T"])
        name = input_params["name"]
        type_of_attack = input_params["attack_mode"]
    except Exception as e:
        print(f"Error in reading the input parameters: {e}")
        return


    # START NODES CREATION FROM THE OLD CODE BASE
    # nodes builder is a function that returns a list of nodes, exploiting the previous code base
    # the decision on the node behavior has to be considered as an hyperparameters set

    if DEBUG:
        print("input_params: ", input_params)

    results_nodes_builder = nodes_builder()

    # list of all the nodes build from the previous code base
    sydelp_nodes: list[Node] = results_nodes_builder[0]
    # shuffle the nodes
    np.random.shuffle(sydelp_nodes)

    attacker: Union[Attacker, None] = results_nodes_builder[1]
    (X_test, y_test) = (results_nodes_builder[4], results_nodes_builder[5])
    metrics_params: dict[str, MetricParams] = results_nodes_builder[3]
    attacker_nodes = attacker.nodes
    # shuffle the attacker nodes
    np.random.shuffle(attacker_nodes)

    # END NODES CREATION FROM THE OLD CODE BASE


    if DEBUG:
        print("attacker: ", attacker)
        print("attacker_nodes: ", attacker_nodes)

    # we will use this node for the DataLogger and VerifierNode
    util_node_copy = sydelp_nodes[0]

    # creating the first four actors (UserInterfaceActor, DataLoggerActor, VerifierNodeActor, AttackerManagerActor), special nodes
    actors.extend(
        [
            UserInterfaceActor.start(0),
            DataLoggerActor.start(1, util_node_copy, X_test, y_test, metrics_params, INIT_MALICIOUS_NODES, HONEST_NODES, name),
            VerifierNodeActor.start(2, util_node_copy, T, exp_num_of_attacks),
            AttackerManagerActor.start(3, attacker_nodes, INIT_MALICIOUS_NODES, OFFSET + HONEST_NODES, ALPHA,
                                       INIT_MALICIOUS_NODES, T, type_of_attack),
        ]
    )

    # creating the actor for the honest participants
    for i in range(OFFSET, HONEST_NODES + OFFSET):
        actors.append(ActiveNodeActor.start(i, sydelp_nodes[i - 3], is_attacker_flag=False, perform_attack=False))

    # creating the actor for the malicious participants
    for i in range(HONEST_NODES + OFFSET, HONEST_NODES + OFFSET + INIT_MALICIOUS_NODES):
        # here we have to specify the type of attack, if independent or colluding
        # indipendent is a specific scenario where the attacker is not colluding with other attackers
        # they start the attack independently
        if type_of_attack == "independent":
            actors.append(ActiveNodeActor.start(i, attacker_nodes[i - (OFFSET + HONEST_NODES)], is_attacker_flag=True,
                                                perform_attack=True))

        # colluding is a specific scenario where the attacker is colluding with other attackers
        # they start to attacker together
        elif type_of_attack == "colluding":
            actors.append(ActiveNodeActor.start(i, attacker_nodes[i - (OFFSET + HONEST_NODES)], is_attacker_flag=True,
                                                perform_attack=False))

    if DEBUG:
        print("Actors created: ", len(actors))
        for i, actor_h in enumerate(actors):
            print(f"Node {i}: {actor_h}")

    # setting partners for all the first actors
    # the parterns are reference to other actors
    # AT HIGH-LEVEL PARTNERS ARE ADDRESSES TO OTHER ACTORS
    for i, actor_h in enumerate(actors[:OFFSET]):
        list_of_partners = [actor for actor in actors[1:] if actor != actor_h]
        try:
            actor_h.tell({"command": "set_partners", "partners": list_of_partners})
            # node_h.tell({"command": "debug"})
        except Exception as e:
            print(f"Failed to send partners to node {i}: {e}")
            return

    # setting partners for all the actors
    for i, actor_h in enumerate(actors[OFFSET:]):
        list_of_partners = [actors[1], actors[2], actors[3]]
        if DEBUG:
            print(f"sending partners for node {i}: {actor_h}")
        try:
            actor_h.tell({"command": "set_partners", "partners": list_of_partners})
            # node_h.tell({"command": "debug"})
        except Exception as e:
            print(f"Failed to send partners to node {i}: {e}")
            return

    # start the interface with UI interaction
    try:
        actors[0].tell({"command": "start_interface", "round": 0})
    except Exception as e:
        print(f"Failed to start the interface: {e}")
        return

    if DEBUG:
        print("Network started.")


if __name__ == "__main__":
    input_file_path = "./params_network/toy_example_input_params.json"
    run_network_implementation(input_file_path)
