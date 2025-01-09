from __future__ import annotations

from actors_nodes.active_node_actor import ActiveNodeActor
from actors_nodes.user_interface_actor import UserInterfaceActor
from actors_nodes.data_logger_actor import DataLoggerActor
from actors_nodes.verifier_node_actor import VerifierNodeActor
from nodes.node import Node

NODES_TYPES = ['node', 'user_interface', 'data_logger']


MAX_NODES = 5

DEBUG = True
def network_implementation(params):
    nodes = []


    if DEBUG:
        print(f"Starting the network, with mock structure")

    nodes.extend(
        [
            UserInterfaceActor.start(0),
            DataLoggerActor.start(1),
            VerifierNodeActor.start(2)
        ]
    )

    ID_START = 3
    for i in range(ID_START, MAX_NODES):
        nodes.append(ActiveNodeActor.start(Node(), i)) # here we start to reuse the code for sydelp
        verifiers = [node for node in nodes if isinstance(node, VerifierNodeActor)]
        verifiers[0].tell({"command": "new_node", "node_id": i, "public_key": "public_key_placeholder", "node": nodes[i]})
        verifiers[0].tell({"command": "DEBUG"})

    for i, node_h in enumerate(nodes):
        list_of_partners = [node for node in nodes if node != node_h]
        print(f"sending partners for node {i}: {node_h}")
        node_h.tell({"command": "set_partner", "partner": list_of_partners})



    # start the interface with UI interaction
    nodes[0].tell({"command": "start_interface"})

if __name__ == "__main__":
    network_implementation(None)