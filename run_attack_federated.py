import os
import utils.split as split
import nodes.node as node

import learning.federated as fl
import utils.initialize as init
import utils.aggregation as agg

from nodes.random_node import RandomNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from keras import models
from numpy.typing import NDArray
from utils.utils import NNumeric

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_rounds: int = 5
n_nodes: int = 15
n_malicious: int = 5
learning_rate: float = 0.001
m: int = 0
mean: float = 0
sd: float = 20
epochs: int = 10
batch_size: int = 10
target_label_1: int = 0
target_label_2: int = 2

X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
X_train, X_test, y_train, y_test = init.iris_data()

global_model: models.Model = init.iris_model(learning_rate)

splits: list[split.Split] = split.class_non_iid_split(X_train,
                                                      y_train,
                                                      n_nodes)

malicious_nodes: list[TargetedLabelFlippingNode] = init.init_nodes(
# malicious_nodes: list[RandomMaliciousNode] = init.init_nodes(
    splits[:n_malicious],
    model_fn=init.iris_model,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    node_class=TargetedLabelFlippingNode,
    target_label_1=target_label_1,
    target_label_2=target_label_2,
    # node_class=RandomMaliciousNode,
    # mean=mean,
    # sd=sd,
)

for node_i in malicious_nodes:
    node_i.attacking = False

nodes: list[node.Node] = init.init_nodes(
    splits[n_malicious:],
    model_fn=init.iris_model,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size
)

nodes = nodes + malicious_nodes

# Create a FederatedLearning instance
federated_learning = fl.FederatedLearning(
    rounds=n_rounds,
    aggregation_function=agg.krum,
    aggregation_params={'m': m},
    nodes=nodes,
    global_model=global_model,
    x_testing=X_test,
    y_testing=y_test
)

federated_learning.start()

federated_learning.metrics
federated_learning.predictions

print("Total running time: %.4f minutes" % federated_learning.execution_time)

federated_learning.save('results/federated_learning', all=True)
