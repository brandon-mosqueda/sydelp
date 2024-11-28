import sys
import os

os.chdir("/home/bmosqueda/doctorado/experiments/decentralized_learning")
sys.path.append("/home/bmosqueda/doctorado/experiments/decentralized_learning")

import networkx as nx

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import norm

import tensorflow as tf
import numpy as np
import os

from numpy.typing import NDArray
from keras.models import Model # type: ignore

import utils.initialize as init
from utils.split import dirichlet_split, Split
from utils.typing import Float, NumArray, KerasModel
from utils.metrics import cos_similarity
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
session = tf.compat.v1.Session(config=config)

local_epoch: int = 10
client_num: int = 100
iterations_num: int = 100
byzantine_num: int = 33
miu: float = 0.1
c_max: float = 0.3
c_min: float = 0.1
alpha: float = 0

# TEST
local_epoch = 2
client_num = 12
iterations_num = 10
byzantine_num = 3
miu = 0.1
c_max = 0.3
c_min = 0.1
alpha = 0

X_train, X_test, y_train, y_test = init.mnist_data()


def get_flatten_weights(model: KerasModel) -> NumArray:
    weights = model.get_weights()
    total_size = sum(w.size for w in weights)

    # Pre-allocate a single array with the necessary size
    flat_weights = np.empty(total_size, dtype="float")

    # Fill the flat_weights array in place
    idx = 0
    for w in weights:
        flat_size = w.size
        flat_weights[idx:idx + flat_size] = w.flatten()
        idx += flat_size

    return flat_weights


def flatten_to_original(weights: NumArray,
                        ref_model: KerasModel) -> list[NumArray]:
    ref_weights: list[NumArray] = ref_model.get_weights()
    new_weights: list[NumArray] = []
    index = 0

    # Directly reshape each section of `weights` without additional slicing
    for layer in ref_weights:
        num_elements = layer.size
        new_weights.append(weights[index:index + num_elements]
                           .reshape(layer.shape))
        index += num_elements

    return new_weights

def set_flatten_weights(model: KerasModel, weights: NumArray) -> None:
    model.set_weights(flatten_to_original(weights, model))

class Client:
    x: NDArray
    y: NDArray
    id: int
    x_num: int
    model: Model
    is_byzantine: bool
    success_prob: float
    fail_prob: float
    momentum: NDArray
    update: NDArray
    seleted_epoch: int
    epochs: int

    def __init__(self, x, y, id, epochs):
        self.x = x
        self.y = y
        self.id = id
        self.epochs = epochs
        self.x_num = len(x)
        self.model = init.mnist_model()
        self.is_byzantine = False
        self.success_prob = 1
        self.fail_prob = 1
        self.seleted_epoch = 0

        total_size: int = sum(w.size for w in self.model.get_weights())
        self.momentum = np.zeros(total_size)
        self.update = np.zeros(total_size)

    def set_weights(self, weights: NDArray):
        set_flatten_weights(self.model, weights)

    def train(self):
        self.model.fit(self.x,
                       self.y,
                       batch_size=64,
                       epochs=self.epochs,
                       verbose=0)

    def set_update(self, global_weights: NDArray) -> None:
        self.update = get_flatten_weights(self.model) - global_weights


def client_beta_selection(clients: list[Client]) -> list[int]:
    selected_ids: list[int] = []

    for client in clients:
        p: float = np.random.beta(client.success_prob, client.fail_prob)

        if p >= 0.9 or (p > 0.2 and np.random.random() < p):
            selected_ids.append(client.id)

    if not selected_ids:
        selected_ids = [c.id for c in clients]

    return selected_ids


def remove_sybils_ids(selected_ids: list[int],
                      clients: list[Client],
                      c_max: float,
                      c_min: float,
                      iteration_num: int) -> list[int]:
    # To detect sybils
    sim_threshold: float = max(c_max * np.exp(-iteration_num / 20), c_min)
    print(f"* Threshold: {sim_threshold}")

    graph: nx.Graph = nx.Graph()
    edges: list[tuple] = []

    for i in range(len(selected_ids)):
        for j in range(i + 1, len(selected_ids)):
            similarity: Float = cos_similarity(
                clients[selected_ids[i]].momentum,
                clients[selected_ids[j]].momentum
            )

            if similarity > sim_threshold:
                edges.append((selected_ids[i], selected_ids[j]))

    graph.add_nodes_from(selected_ids)
    graph.add_edges_from(edges)

    max_connected_component: set = sorted(nx.connected_components(graph),
                                          key=len,
                                          reverse=True)[0]

    # Remove potential sybils
    remove_vals: list[int] = []
    if (len(max_connected_component) > 1):
        for i in max_connected_component:
            clients[i].fail_prob += 1
            remove_vals.append(i)

    if len(remove_vals) < len(selected_ids) - 1:
        for val in remove_vals:
            selected_ids.remove(val)

    return selected_ids


# Little Is Enough (LIE) attack
def LIE(clients: list[Client], selected_ids: list[int]) -> None:
    benign_ids: list[int] = [i for i in selected_ids
                             if not clients[i].is_byzantine]

    s: int = int(client_num/2 + 1) - byzantine_num

    z: NDArray = norm.ppf((client_num - byzantine_num - s)
                          / (client_num - byzantine_num))
    mean: NDArray = np.mean([clients[i].update for i in benign_ids], axis=0)
    std: NDArray = np.std([clients[i].update for i in benign_ids], axis=0)

    bad_update: NDArray = mean - z * std

    for i in selected_ids:
        if clients[i].is_byzantine:
            clients[i].update = bad_update


def MAB_FL(clients: list[Client],
           selected_ids: list[int],
           iteration_num: int,
           c_max: float,
           c_min: float) -> tuple[np.float_, NDArray]:
    for i in selected_ids:
        clients[i].momentum = (clients[i].update
                               + miu**(iteration_num - clients[i].seleted_epoch)
                               * clients[i].momentum)

        clients[i].momentum = (clients[i].momentum
                               / np.linalg.norm(clients[i].momentum))

        clients[i].seleted_epoch = iteration_num

    selected_ids = remove_sybils_ids(selected_ids,
                                     clients,
                                     c_max=c_max,
                                     c_min=c_min,
                                     iteration_num=iter)

    local_updates: NDArray = np.array(
        [clients[i].momentum for i in selected_ids])

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(local_updates)

    estimator = AgglomerativeClustering(2)
    estimator.fit(X_reduced)
    label_pred = estimator.labels_

    ids_cluster1: list[int] = []
    ids_cluster2: list[int] = []
    for i in range(len(selected_ids)):
        if label_pred[i] == 0:
            ids_cluster1.append(selected_ids[i])
        else:
            ids_cluster2.append(selected_ids[i])

    mean_cluster1: NDArray = np.mean(
        [clients[i].momentum for i in ids_cluster1],
        axis=0)
    mean_cluster2: NDArray = np.mean(
        [clients[i].momentum for i in ids_cluster2],
        axis=0)

    cos_between_clusters: Float = cos_similarity(mean_cluster1,
                                                 mean_cluster2)

    if cos_between_clusters < alpha:
        smallest_cluster = (ids_cluster2
                           if len(ids_cluster1) > len(ids_cluster2)
                           else ids_cluster1)

        for i in smallest_cluster:
            clients[i].fail_prob += 1
            selected_ids.remove(i)

    # Increase the prob for the remaining clients
    for i in selected_ids:
        clients[i].success_prob += 1

    print("* Final aggregation:", selected_ids)

    lr: np.float_ = np.median(
        [np.linalg.norm(clients[i].update) for i in selected_ids])

    return lr, np.mean([clients[i].momentum for i in selected_ids], axis=0)


model_accuracy_list: list[float] = []
model_loss_list: list[float] = []

global_model: Model = init.mnist_model()
global_weights: NDArray = get_flatten_weights(global_model)

splits: list[Split] = dirichlet_split(
    X_train,
    y_train,
    n_splits=client_num,
    alpha=0.5,
    split_min_size=16,
    seed=2
)

clients: list[Client] = []
for i in range(0, client_num):
    clients.append(Client(splits[i]['X'], splits[i]['y'], i, local_epoch))

    if i >= client_num - byzantine_num:
        clients[i].is_byzantine = True

for c in clients:
    c.set_weights(global_weights)

iter = 1
for iter in range(iterations_num):
    print(f"\n\n*** Iteration {iter} ***")
    selected_ids: list[int]

    if iter < 10:
        selected_ids = [i for i in range(client_num)]
    else:
        selected_ids = client_beta_selection(clients)

    for i in selected_ids:
        if not clients[i].is_byzantine:
            print(f"\t + Training on client {clients[i].id}")

            clients[i].train()
            clients[i].set_update(global_weights)

    LIE(clients, selected_ids)
    lr, global_update = MAB_FL(clients,
                               selected_ids,
                               iteration_num=iter,
                               c_max=c_max,
                               c_min=c_min)

    global_weights = global_weights + lr * global_update
    set_flatten_weights(global_model, global_weights)

    test_loss: float = global_model.evaluate(X_test, y_test, verbose=0)
    preds = global_model.predict(X_test, verbose=0)
    preds = np.argmax(preds, axis=1)
    test_acc = float(accuracy_score(y_test, preds))
    print("* Loss: %.4f, Accuracy: %.4f" % (test_loss, test_acc))

    model_accuracy_list.append(test_acc)
    model_loss_list.append(test_loss)

    for c in clients:
        c.set_weights(global_weights)
