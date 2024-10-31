import os

import learning.federated as fl
import utils.initialize as init
import utils.aggregation as agg

from importlib import reload
from sklearn.metrics import f1_score, cohen_kappa_score

fl = reload(fl)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# initializer = init.init_mnist(nodes_num=100)
initializer = init.init_iris(nodes_num=5)

# Create a FederatedLearning instance
federated_learning = fl.FederatedLearning(
    rounds=5,
    aggregation_params={
        'function': agg.krum,
        'params': {'m': 5}
    },
    metrics_params={
        'kappa': {'function': cohen_kappa_score, 'params': {}},
        'f1': {'function': f1_score, 'params': {'average': "micro"}}
    },
    attack_metrics_params={
        'f1': {'function': f1_score, 'params': {'average': "micro"}}
    },
    nodes=initializer['nodes'],
    global_model=initializer['global_model'],
    x_testing=initializer['X_test'],
    y_testing=initializer['y_test']
)

federated_learning.start()

federated_learning.metrics
federated_learning.attack_metrics
federated_learning.predictions

print("Total running time: %.4f minutes" % federated_learning.execution_time)

federated_learning.save('trash/federated_learning', all=True)
