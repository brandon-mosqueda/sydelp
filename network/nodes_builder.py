"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2025-01-21 18:41:51
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-02-05 11:13:44
FilePath: network/nodes_builder.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
import sys
import os

import utils.initialize as init
import utils.utils as utils

from typing import Union
from nodes.node import Node
from utils.utils import IntArray, as_name, NumArray
from attack.attacker import Attacker
from learning.learning import Learning
from learning.learning import MetricParams
from keras.src.models import Model as KerasModel

DEBUG = False

def nodes_builder() -> list[list[Node], Union[Attacker, None], KerasModel, dict[str, MetricParams], NumArray, IntArray]:
    # we are going to use the same structure as in run.py to get the nodes
    # TODO roadmap:
    # 1. get the params from the file
    # 2. get the dataset
    # 3. get the model
    # 4. get the metrics
    # 5. build the various nodes in the network
    # 6. get the attacker
    # 7. return the nodes, attacker, global_model, metrics_params
    params_file: str = 'params/params/32.sydelp_mnist_random.json'
    params: dict = utils.read_json(params_file)
    
    if DEBUG:
        print(params)
        print("Getting dataset...")

    X_train: NumArray; X_test: NumArray; X_mal: NumArray;
    y_train: IntArray; y_test: IntArray; y_mal: IntArray;  
    X_train, X_test, y_train, y_test = init.get_dataset(params)
    
    if DEBUG:
        print("dataset loaded")
    
    if DEBUG:
        print("Getting model...")

    global_model: KerasModel = init.get_model_by_dataset(params)
    models: list[KerasModel] = utils.replicate_model(global_model,
                                                    n=params['nodes_num'])
    
    if DEBUG:
        print("Getting metrics...")

    metrics_params: dict[str, MetricParams] = init.get_metrics(params)
    
    # nodes only gets the training dataset, the model and the metrics
    nodes: list[Node] = init.get_nodes_by_protocol(params,
                                                    X_train=X_train,
                                                    y_train=y_train,
                                                    models=models)

    attacker: Union[Attacker, None] = init.get_attacker(nodes, params)

    return [nodes, attacker, global_model, metrics_params, X_test, y_test]

def test_nodes_builder():
    print("Testing nodes_builder...")
    
    nodes, attacker, global_model, metrics_params = nodes_builder()
    # assert len(nodes) == 5
    # assert attacker is not None
    # assert global_model is not None
    # assert len(metrics_params) == 3
    # print("All tests passed.")
    
    print("Nodes:")
    for node in nodes:
        print(node)
    print("Attacker:")
    print(attacker)
    print("Global Model:")
    print(global_model.get_weights())
    print("Metrics Params:")
    print(metrics_params)
    
    
    
    
# if __name__ == "__main__":
#     test_nodes_builder()