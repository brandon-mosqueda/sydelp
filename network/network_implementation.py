from __future__ import annotations

from actors_nodes.active_node_actor import ActiveNodeActor
from actors_nodes.user_interface_actor import UserInterfaceActor
from actors_nodes.data_logger_actor import DataLoggerActor
from actors_nodes.verifier_node_actor import VerifierNodeActor
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

NODES_TYPES = ['node', 'user_interface', 'data_logger']


MAX_NODES = 5

def get_nodes() -> None:
    # we are going to use the same structure as in run.py to get the nodes
    params_file: str = 'params/26.sydelp_spam_random.test.json'
    params: dict = utils.read_json(params_file)

    X_train: NumArray; X_test: NumArray; X_mal: NumArray;
    y_train: IntArray; y_test: IntArray; y_mal: IntArray;
    X_train, X_test, y_train, y_test = init.get_dataset(params)

    global_model: KerasModel = init.get_model_by_dataset(params)
    models: list[KerasModel] = utils.replicate_model(global_model,
                                                    n=params['nodes_num'])

    metrics_params: dict[str, MetricParams] = init.get_metrics(params)
    nodes: list[Node] = init.get_nodes_by_protocol(params,
                                                    X_train=X_train,
                                                    y_train=y_train,
                                                    models=models)

    attacker: Union[Attacker, None] = init.get_attacker(nodes, params)

    return (nodes, attacker, global_model, metrics_params)


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

    # first try to get the nodes from the params
    nodes, attacker, global_model, metrics_params = get_nodes()

    print(f"Nodes: {nodes}")
    print(f"Attacker: {attacker}")
    print(f"Global Model: {global_model}")
    print(f"Metrics Params: {metrics_params}")
    exit()

    for i, node_h in enumerate(nodes):
        list_of_partners = [node for node in nodes if node != node_h]
        print(f"sending partners for node {i}: {node_h}")
        node_h.tell({"command": "set_partner", "partner": list_of_partners})



    # start the interface with UI interaction
    nodes[0].tell({"command": "start_interface", "round": 0})

if __name__ == "__main__":
    network_implementation(None)