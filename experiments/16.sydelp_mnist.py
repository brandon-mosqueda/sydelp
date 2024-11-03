import sys
import os

os.chdir("/home/bmosqueda/doctorado/experiments/decentralized_learning")
sys.path.append("/home/bmosqueda/doctorado/experiments/decentralized_learning")

import utils.initialize as init
import utils.utils as utils

from nodes.node import Node
from numpy.typing import NDArray
from utils.utils import NNumeric
from utils.split import dirichlet_split, Split
from utils.aggregation import krum
from learning.federated import FederatedLearning
from utils.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.src.models import Model as KerasModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params: dict = utils.read_json('params/16.sydelp_mnist.json')

X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
X_train, X_test, y_train, y_test = init.mnist_data()

splits: list[Split] = dirichlet_split(
    X_train,
    y_train,
    n_splits=params['nodes_num'],
    alpha=params['alpha'],
    split_min_size=params['split_min_size'],
)

global_model: KerasModel = init.mnist_model(
    learning_rate = params['learning_rate'],
    dense_units = params['dense_units'],
)

models: list[KerasModel] = utils.replicate_model(global_model,
                                                 n=params['nodes_num'])

nodes: list[Node] = []

for i, split in enumerate(splits):
    nodes.append(Node(x=split['X'],
                      y=split['y'],
                      model=models[i],
                      epochs=params['local_epochs_num'],
                      batch_size=params['batch_size']))

# Create a FederatedLearning instance
federated_learning = FederatedLearning(
    rounds=params['iterations_num'],
    nodes=nodes,
    global_model=global_model,
    x_testing=X_test,
    y_testing=y_test,
    aggregation_params={
        'function': krum,
        'params': {'m': params['expected_malicious_num']}
    },
    metrics_params={'accuracy': {'function': accuracy_score, 'params': {}}}
)

federated_learning.start()

print("Total running time: %.4f minutes" % federated_learning.execution_time)

results_dir: str = os.path.join(
    params['results_dir'], "sydelp", "mnist", "no_attack")

metadata = {
    'Protocol': 'SyDeLP',
    'Dataset': 'MNIST',
    'Attack': 'No attack'
}
federated_learning.save(results_dir, all=True, metadata=metadata)
