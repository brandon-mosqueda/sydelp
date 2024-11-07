import sys
import os

os.chdir("/home/bmosqueda/doctorado/experiments/decentralized_learning")
sys.path.append("/home/bmosqueda/doctorado/experiments/decentralized_learning")

import utils.initialize as init
import utils.utils as utils

from nodes.node import Node
from nodes.random_node import RandomNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.sign_flipping_node import SignFlippingNode
from numpy.typing import NDArray
from utils.utils import NNumeric, as_name
from utils.split import dirichlet_split, Split
from learning.federated import FederatedLearning, AggParams, MetricParams
from keras.src.models import Model as KerasModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params_file: str = sys.argv[1]
# params_file: str = 'params/11.dl_spam.test.json'
params: dict = utils.read_json(params_file)

X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
X_train, X_test, y_train, y_test = init.get_dataset(params)

aggregation_params: AggParams = init.get_aggregation_by_protocol(params)
metrics_params: dict[str, MetricParams] = init.get_metrics(params)

splits: list[Split] = dirichlet_split(
    X_train,
    y_train,
    n_splits=params['nodes_num'],
    alpha=params['alpha'],
    split_min_size=params['split_min_size'],
    seed=params['seed'],
)

global_model: KerasModel = init.get_model_by_dataset(params)

models: list[KerasModel] = utils.replicate_model(global_model,
                                                 n=params['nodes_num'])

nodes: list[Node] = []

counter = 0
for i in range(params.get('random_malicious_num', 0)):
    nodes.append(RandomNode(mean=params['attack_mean'],
                            sd=params['attack_sd'],
                            x=splits[i]['X'],
                            y=splits[i]['y'],
                            model=models[i],
                            epochs=params['local_epochs_num'],
                            batch_size=params['batch_size']))
    counter += 1

for i in range(counter, counter + params.get('sign_flip_malicious_num', 0)):
    nodes.append(SignFlippingNode(
        scale_factor=params['attack_scale_factor'],
        x=splits[i]['X'],
        y=splits[i]['y'],
        model=models[i],
        epochs=params['local_epochs_num'],
        batch_size=params['batch_size']
    ))
    counter += 1

for i in range(counter, counter + params.get('label_flip_malicious_num', 0)):
    nodes.append(TargetedLabelFlippingNode(
        source=params['source_label'],
        target=params['target_label'],
        x=splits[i]['X'],
        y=splits[i]['y'],
        model=models[i],
        epochs=params['local_epochs_num'],
        batch_size=params['batch_size']
    ))
    counter += 1

for i in range(counter, params['nodes_num']):
    nodes.append(Node(x=splits[i]['X'],
                      y=splits[i]['y'],
                      model=models[i],
                      epochs=params['local_epochs_num'],
                      batch_size=params['batch_size']))
    counter += 1

# Create a FederatedLearning instance
federated_learning = FederatedLearning(
    rounds=params['iterations_num'],
    nodes=nodes,
    global_model=global_model,
    x_testing=X_test,
    y_testing=y_test,
    aggregation_params=aggregation_params,
    metrics_params=metrics_params,
)

federated_learning.start()
print("Total running time: %.4f minutes" % federated_learning.execution_time)

results_dir: str = os.path.join(
    params['results_dir'],
    as_name(params['protocol']),
    as_name(params['dataset']),
    as_name(params['attack']),
    as_name(params['seed']),
)

metadata = {
    'Protocol': params['protocol'],
    'Dataset': params['dataset'],
    'Attack': params['attack'],
    'Seed': params['seed']
}
federated_learning.save(results_dir, metadata=metadata)
