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
from utils.utils import IntArray, as_name, NumArray
from utils.split import dirichlet_split, Split, balanced_split
from learning.learning import Learning
from learning.learning import MetricParams
from keras.src.models import Model as KerasModel
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params_file: str = sys.argv[1]
# params_file: str = 'params/26.sydelp_spam_random.test.json'
# params_file: str = 'params/15.dl_spam_label_flip.test.json'
params: dict = utils.read_json(params_file)
print(params)

X_train: NumArray; X_test: NumArray; X_mal: NumArray;
y_train: IntArray; y_test: IntArray; y_mal: IntArray;
X_train, X_test, y_train, y_test = init.get_dataset(params)

global_model: KerasModel = init.get_model_by_dataset(params)
models: list[KerasModel] = utils.replicate_model(global_model,
                                                 n=params['nodes_num'])

metrics_params: dict[str, MetricParams] = init.get_metrics(params)

malicious_num: int = (
    params.get('random_malicious_num', 0)
    + params.get('sign_flip_malicious_num', 0)
    + params.get('label_flip_malicious_num', 0)
)
honest_num: int = params['nodes_num'] - malicious_num

nodes: list[Node] = []

if malicious_num > 0:
    # Divide the training set for malicious users proportionally
    X_train, X_mal, y_train, y_mal = train_test_split(
        X_train,
        y_train,
        stratify=y_train,
        test_size=malicious_num/params['nodes_num']
    )

    mal_splits: list[Split] = balanced_split(
        X_mal,
        y_mal,
        n_splits=malicious_num,
        seed=params['seed'],
    )

    for _ in range(params.get('random_malicious_num', 0)):
        split: Split = mal_splits.pop()

        nodes.append(RandomNode(
            mean=params['attack_mean'],
            sd=params['attack_sd'],
            x=split['X'],
            y=split['y'],
            model=models.pop(),
            epochs=params['local_epochs_num'],
            batch_size=params['batch_size']
        ))

    for _ in range(params.get('sign_flip_malicious_num', 0)):
        split: Split = mal_splits.pop()

        nodes.append(SignFlippingNode(
            scale_factor=params['attack_scale_factor'],
            x=split['X'],
            y=split['y'],
            model=models.pop(),
            epochs=params['local_epochs_num'],
            batch_size=params['batch_size']
        ))

    for _ in range(params.get('label_flip_malicious_num', 0)):
        split: Split = mal_splits.pop()

        nodes.append(TargetedLabelFlippingNode(
            source=params['source_label'],
            target=params['target_label'],
            x=split['X'],
            y=split['y'],
            model=models.pop(),
            epochs=params['local_epochs_num'],
            batch_size=params['batch_size']
        ))

hon_splits: list[Split] = dirichlet_split(
    X_train,
    y_train,
    n_splits=honest_num,
    alpha=params['alpha'],
    split_min_size=params['split_min_size'],
    seed=params['seed'],
)

for _ in range(honest_num):
    split: Split = hon_splits.pop()

    nodes.append(Node(x=split['X'],
                      y=split['y'],
                      model=models.pop(),
                      epochs=params['local_epochs_num'],
                      batch_size=params['batch_size']))

# Create a FederatedLearning instance
learning_controller: Learning = init.get_controller_by_protocol(
    params=params,
    nodes=nodes,
    global_model=global_model,
    x_testing=X_test,
    y_testing=y_test,
    metrics_params=metrics_params,
)

learning_controller.start()
print("Total running time: %.4f minutes" % learning_controller.execution_time)

results_dir: str = os.path.join(
    params['results_dir'],
    as_name(params['protocol']),
    as_name(params['dataset']),
    as_name(params['attack']),
    as_name(params['seed']),
    as_name(params['weighting_mode']),
)

metadata = {
    'Protocol': params['protocol'],
    'Dataset': params['dataset'],
    'Attack': params['attack'],
    'Seed': params['seed'],
    "WeightingMode": params['weighting_mode'],
}
learning_controller.save(results_dir, metadata=metadata)
