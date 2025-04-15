import sys
import os
import random
import numpy as np
import tensorflow as tf

import utils.initialize as init
import utils.utils as utils

from typing import Union
from nodes.node import Node
from utils.utils import IntArray, NumArray
from attack.attacker import Attacker
from learning.learning import Learning
from learning.learning import MetricParams
from keras.src.models import Model as KerasModel
from utils.default_values import fill_with_defaults

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dataset_config_file: str = sys.argv[1]
setup_config_file: str = sys.argv[2]
seed: int = int(sys.argv[3])

params: dict = utils.read_json(dataset_config_file)
setup_params: dict = utils.read_json(setup_config_file)

for key, value in setup_params.items():
    params[key] = value

params = fill_with_defaults(params)
params['seed'] = seed
print(params)

# Set random seed for Python's built-in random module
random.seed(params['seed'])
# This also works for scikit-learn
np.random.seed(params['seed'])
tf.random.set_seed(params['seed'])

X_train: NumArray; X_test: NumArray; X_mal: NumArray;
y_train: IntArray; y_test: IntArray; y_mal: IntArray;
X_train, X_test, y_train, y_test = init.get_dataset(params)

global_model: KerasModel = init.get_model_by_dataset(params)
models: list[KerasModel] = utils.replicate_model(global_model,
                                                 n=params['nodes_num'])

metrics_params: dict[str, MetricParams] = init.get_metrics(params)
nodes: list[Node] =  init.get_nodes_by_protocol(params,
                                                X_train=X_train,
                                                y_train=y_train,
                                                models=models)

attacker: Union[Attacker, None] = init.get_attacker(nodes, params)

# Create a FederatedLearning instance
learning_controller: Learning = init.get_controller_by_protocol(
    params=params,
    nodes=nodes,
    global_model=global_model,
    x_testing=X_test,
    y_testing=y_test,
    metrics_params=metrics_params,
    attacker=attacker
)

learning_controller.start()
print("Total running time: %.4f minutes" % learning_controller.execution_time)

learning_controller.save(
    init.get_results_dir(params),
    metadata=init.get_metadata(params)
)
