import os

import federated_learning as fl
import initialize as init

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# initializer = init.init_mnist(nodes_num=100)
initializer = init.init_iris(nodes_num=15)

# Create a FederatedLearning instance
federated_learning = fl.FederatedLearning(
    rounds=5,
    nodes=initializer['nodes'],
    global_model=initializer['global_model'],
    x_testing=initializer['X_test'],
    y_testing=initializer['y_test']
)

federated_learning.start()

federated_learning.metrics
federated_learning.predictions

print("Total running time: %.4f minutes" % federated_learning.execution_time)

federated_learning.save('results/federated_learning', all=True)
