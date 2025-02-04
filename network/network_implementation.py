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

# hyoerparameters
HONEST_NODES = 20 # check this parameter with the actual dataset parameters
INIT_MALICIOUS_NODES = 15 # check this parameter with the actual dataset parameters, it is equal to MAX_COMPUTING_POWER
OFFSET = 4
ALPHA = 2

def read_input_params():
    # read the input parameters from ./params_network/1_input_params.json
    try:
        with open("./params_network/15_input_params.json", "r") as f:
            input_params = json.load(f)
    except Exception as e:
        print(f"Error in reading the input parameters: {e}")
        return None

    return input_params

DEBUG = False
def network_implementation(params):
    nodes = []

    if DEBUG:
        print(f"Starting the network, with mock structure")

    try:
        input_params = read_input_params()
        HONEST_NODES = input_params["HONEST_NODES"]
        INIT_MALICIOUS_NODES = input_params["INIT_MALICIOUS_NODES"]
        ALPHA = float(input_params["alpha"])
        exp_num_of_attacks = float(input_params["exp_num_of_attackers"])
        T = float(input_params["T"])
        name = input_params["name"]
    except Exception as e:
        print(f"Error in reading the input parameters: {e}")
        return




    print("input_params: ", input_params)


    results_nodes_builder = nodes_builder()

    sydelp_nodes: list[Node] = results_nodes_builder[0]
    # shuffle the nodes
    np.random.shuffle(sydelp_nodes)

    attacker: Union[Attacker, None] = results_nodes_builder[1]
    (X_test, y_test) = (results_nodes_builder[4], results_nodes_builder[5])
    metrics_params: dict[str, MetricParams] = results_nodes_builder[3]
    attacker_nodes = attacker.nodes
    # shuffle the attacker nodes
    np.random.shuffle(attacker_nodes)

    print("attacker: ", attacker)
    print("attacker_nodes: ", attacker_nodes)

    util_node_copy = sydelp_nodes[0]

    nodes.extend(
        [
            UserInterfaceActor.start(0),
            DataLoggerActor.start(1, util_node_copy, X_test, y_test, metrics_params, INIT_MALICIOUS_NODES, HONEST_NODES, name),
            VerifierNodeActor.start(2, util_node_copy, T, exp_num_of_attacks),
            AttackerManagerActor.start(3, attacker_nodes, INIT_MALICIOUS_NODES, OFFSET+HONEST_NODES, ALPHA, INIT_MALICIOUS_NODES, T),
        ]
    )


    for i in range(OFFSET, HONEST_NODES+OFFSET):
        nodes.append(ActiveNodeActor.start(i, sydelp_nodes[i-3], is_attacker_flag=False))

    for i in range(HONEST_NODES+OFFSET, HONEST_NODES+OFFSET+INIT_MALICIOUS_NODES):
        nodes.append(ActiveNodeActor.start(i, attacker_nodes[i-(OFFSET+HONEST_NODES)], is_attacker_flag=True))


    for i, node_h in enumerate(nodes[:OFFSET]):
        list_of_partners = [node for node in nodes[1:] if node != node_h]
        try:
            node_h.tell({"command": "set_partners", "partners": list_of_partners})
            # node_h.tell({"command": "debug"})
        except Exception as e:
            print(f"Failed to send partners to node {i}: {e}")
            return

    for i, node_h in enumerate(nodes[OFFSET:]):
        list_of_partners = [nodes[1], nodes[2], nodes[3]]
        print(f"sending partners for node {i}: {node_h}")
        try:
            node_h.tell({"command": "set_partners", "partners": list_of_partners})
            # node_h.tell({"command": "debug"})
        except Exception as e:
            print(f"Failed to send partners to node {i}: {e}")
            return



    # start the interface with UI interaction
    try:
        nodes[0].tell({"command": "start_interface", "round": 0})
    except Exception as e:
        print(f"Failed to start the interface: {e}")
        return

    print("Network started.")

if __name__ == "__main__":
    network_implementation(None)