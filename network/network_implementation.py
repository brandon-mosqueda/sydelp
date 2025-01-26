from __future__ import annotations

from actors_nodes.active_node_actor import ActiveNodeActor
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

# hyoerparameters
HONEST_NODES = 5
INIT_MALICIOUS_NODES = 3


DEBUG = False
def network_implementation(params):
    nodes = []


    if DEBUG:
        print(f"Starting the network, with mock structure")


    results_nodes_builder = nodes_builder()

    sydelp_nodes: list[Node] = results_nodes_builder[0]
    attacker: Union[Attacker, None] = results_nodes_builder[1]
    (X_test, y_test) = (results_nodes_builder[4], results_nodes_builder[5])
    metrics_params: dict[str, MetricParams] = results_nodes_builder[3]
    attacker_nodes = attacker.nodes

    print("attacker: ", attacker)
    print("attacker_nodes: ", attacker_nodes)

    util_node_copy = sydelp_nodes[0]

    nodes.extend(
        [
            UserInterfaceActor.start(0),
            DataLoggerActor.start(1, util_node_copy, X_test, y_test, metrics_params),
            VerifierNodeActor.start(2, util_node_copy)
        ]
    )

    OFFSET = 3
    for i in range(OFFSET, HONEST_NODES+OFFSET):
        nodes.append(ActiveNodeActor.start(i, sydelp_nodes[i-3], is_attacker_flag=False))

    for i in range(HONEST_NODES+OFFSET, HONEST_NODES+OFFSET+INIT_MALICIOUS_NODES):
        nodes.append(ActiveNodeActor.start(i, attacker_nodes[i-(OFFSET+HONEST_NODES)], is_attacker_flag=True))


    for i, node_h in enumerate(nodes[:3]):
        list_of_partners = [node for node in nodes[1:] if node != node_h]
        try:
            node_h.tell({"command": "set_partners", "partners": list_of_partners})
            # node_h.tell({"command": "debug"})
        except Exception as e:
            print(f"Failed to send partners to node {i}: {e}")
            return

    for i, node_h in enumerate(nodes[3:]):
        list_of_partners = [nodes[1], nodes[2]]
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