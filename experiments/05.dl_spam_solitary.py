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
from utils.utils import NNumeric
from utils.split import dirichlet_split, Split
from learning.federated import FederatedLearning
from utils.metrics import f1_score, label_flipping_success_rate, label_recall
from sklearn.metrics import accuracy_score
from keras.src.models import Model as KerasModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params: dict = utils.read_json('params/05.dl_spam_solitary.json')

X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
X_train, X_test, y_train, y_test = init.spam_data(
    testing_proportion=params['testing_proportion'],
    vocabulary_size=params['vocabulary_size'],
    max_sequence_length=params['max_sequence_length'],
)

splits: list[Split] = dirichlet_split(
    X_train,
    y_train,
    n_splits=params['nodes_num'],
    alpha=params['alpha'],
    split_min_size=params['split_min_size'],
)

global_model: KerasModel = init.spam_model(
    learning_rate = params['learning_rate'],
    vocabulary_size = params['vocabulary_size'],
    sequence_length=params['max_sequence_length'],
    embedding_dim = params['embedding_dim'],
    lstm_units = params['lstm_units'],
)

models: list[KerasModel] = utils.replicate_model(global_model,
                                                 n=params['nodes_num'])

nodes: list[Node] = []

counter = 0
for i in range(params['random_malicious_num']):
    nodes.append(RandomNode(mean=params['attack_mean'],
                            sd=params['attack_sd'],
                            x=splits[i]['X'],
                            y=splits[i]['y'],
                            model=models[i],
                            epochs=params['local_epochs_num'],
                            batch_size=params['batch_size']))
    counter += 1

for i in range(counter, counter + params['sign_flip_malicious_num']):
    nodes.append(SignFlippingNode(
        scale_factor=params['attack_scale_factor'],
        x=splits[i]['X'],
        y=splits[i]['y'],
        model=models[i],
        epochs=params['local_epochs_num'],
        batch_size=params['batch_size']
    ))
    counter += 1

for i in range(counter, counter + params['label_flip_malicious_num']):
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
    metrics_params={
        'accuracy': {'function': accuracy_score, 'params': {}},
        'f1_score': {'function': f1_score, 'params': {}}
    },
    attack_metrics_params={
        'attack_success_rate': {
            'function': label_flipping_success_rate,
            'params': {
                'source': params['source_label'],
                'target': params['target_label']
            }
        },
        'label_recall': {
            'function': label_recall,
            'params': {'label': params['source_label']}
        }
    }
)

federated_learning.start()

print("Total running time: %.4f minutes" % federated_learning.execution_time)

results_dir: str = os.path.join(
    params['results_dir'], "DL", "spam", "solitary")

metadata = {
    'Protocol': 'DL',
    'Dataset': 'SMS Spam',
    'Attack': 'Solitary'
}
federated_learning.save(results_dir, all=True, metadata=metadata)
