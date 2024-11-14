import networkx as nx

from sklearn.decomposition import PCA
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering

import tensorflow as tf
import numpy as np
import os

from numpy.typing import NDArray
from keras.models import Model # type: ignore

import utils.initialize as init
from utils.split import dirichlet_split, Split
from utils.utils import remove_indices
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
session = tf.compat.v1.Session(config=config)

local_epoch: int = 10
client_num: int = 100
iterations_num: int = 100
f: int = 33
miu: float = 0.1
c_max: float = 0.3
c_min: float = 0.1
alpha: float = 0

# TEST
local_epoch = 2
client_num = 12
iterations_num = 10
f = 3
miu = 0.1
c_max = 0.3
c_min = 0.1
alpha = 0

X_train, X_test, y_train, y_test = init.mnist_data()


def get_flatten_weights(model: Model) -> NDArray:
    weights: list[NDArray] = model.get_weights()

    return np.concatenate([w.flatten() for w in weights], dtype="float")


def set_flatten_weights(model: Model, weights: NDArray) -> None:
    original_weights: list[NDArray] = model.get_weights()

    new_weights: list = []
    index: int = 0

    for w in original_weights:
        # Calculate the number of elements in this layer's weights
        num_elements = w.size

        # Reshape and add to the new weights list
        new_weights.append(weights[index:index + num_elements].reshape(w.shape))

        index += num_elements

    # Set the reshaped weights back to the model
    model.set_weights(new_weights)


class Client:
    update: NDArray
    X: NDArray
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

    def __init__(self, X, y, id):
        self.X = X
        self.y = y
        self.id = id
        self.x_num = len(X)
        self.model = init.mnist_model()
        self.is_byzantine = False
        self.success_prob = 1
        self.fail_prob = 1
        self.momentum = np.array(get_flatten_weights(self.model)) * 0
        self.update = np.array(get_flatten_weights(self.model)) * 0
        self.seleted_epoch = 0

    def set_weights(self, weights: NDArray):
        set_flatten_weights(self.model, weights)

    def train(self):
        self.model.fit(self.X,
                       self.y,
                       batch_size=64,
                       epochs=local_epoch,
                       verbose=0)

        self.update = get_flatten_weights(self.model) - global_weights


def cos_similarity(x: NDArray, y: NDArray, unit: bool) -> np.float32:
    res: np.float32 = (x * y).sum()

    if not unit:
        res /= np.linalg.norm(x) * np.linalg.norm(y)

    return np.float32(min(res, 1))


def LIE(clients: list[Client], selectedId: list[int]) -> None:
    benignId: list[int] = []

    for i in selectedId:
        if clients[i].is_byzantine == False:
            benignId.append(i)

    s: int = int(client_num/2 + 1) - f

    z: NDArray = norm.ppf((client_num - f - s) / (client_num - f))
    mean: NDArray = np.mean([clients[i].update for i in benignId], axis=0)
    std: NDArray = np.std([clients[i].update for i in benignId], axis=0)

    bad_update: NDArray = mean - z * std

    for i in selectedId:
        if clients[i].is_byzantine == True:
            clients[i].update = bad_update


def TS(clients: list[Client]) -> list[int]:
    selectedId: list[int] = []

    for i in range(client_num):
        p: float = np.random.beta(clients[i].success_prob, clients[i].fail_prob)

        if p >= 0.9:
            selectedId.append(clients[i].id)
        if p > 0.2 and p <= 0.9 and np.random.random() < p:
            selectedId.append(clients[i].id)

    if len(selectedId) == 0:
        selectedId = [i for i in range(client_num)]

    return selectedId


def MAB_FL(clients: list[Client],
           selectedId: list[int],
           iter: int) -> tuple[np.float32, NDArray]:
    for i in selectedId:
        clients[i].momentum = (clients[i].update
                               + miu**(iter - clients[i].seleted_epoch)
                               * clients[i].momentum)

        clients[i].momentum = (clients[i].momentum
                               / np.linalg.norm(clients[i].momentum))

        clients[i].seleted_epoch = iter

    G: nx.Graph = nx.Graph()
    edges: list[tuple] = []

    # To detect sybils
    sim_threshold: float = max(c_max * np.exp(-iter / 20), c_min)
    print(f"* Threshold: {sim_threshold}")

    for i in range(len(selectedId)):
        for j in range(i+1, len(selectedId)):
            similarity: np.float32 = cos_similarity(
                clients[selectedId[i]].momentum,
                clients[selectedId[j]].momentum,
                unit=True
            )

            if similarity > sim_threshold:
                edges.append((selectedId[i], selectedId[j]))

    G.add_nodes_from(selectedId)
    G.add_edges_from(edges)

    C: list = sorted(nx.connected_components(G), key=len, reverse=True)

    # Remove sybils
    remove_ids: list[int] = []
    if (len(C[0]) > 1):
        for i in C[0]:
            clients[i].fail_prob += 1
            remove_ids.append(i)

    if len(remove_ids) < len(selectedId) - 1:
        selectedId = remove_indices(selectedId, remove_ids)
        print(f"* Selected Ids: {selectedId}")
    else:
        print("* Just one model was selected as non-sybil, take them all")

    local_updates: NDArray = np.array([clients[i].momentum for i in selectedId])

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(local_updates)

    estimator = AgglomerativeClustering(2)
    estimator.fit(X_reduced)
    label_pred = estimator.labels_

    selectedId_c1: list[int] = []
    selectedId_c2: list[int] = []
    for i in range(len(selectedId)):
        if label_pred[i] == 0:
            selectedId_c1.append(selectedId[i])
        else:
            selectedId_c2.append(selectedId[i])

    m1: NDArray = np.mean([clients[i].momentum for i in selectedId_c1], axis=0)
    m2: NDArray = np.mean([clients[i].momentum for i in selectedId_c2], axis=0)

    cos_between_clusters = cos_similarity(m1, m2, False)

    if cos_between_clusters < alpha:
        if len(selectedId_c1) > len(selectedId_c2):
            for i in selectedId_c2:
                clients[i].fail_prob += 1
                selectedId.remove(i)
        else:
            for i in selectedId_c1:
                clients[i].fail_prob += 1
                selectedId.remove(i)

    for i in selectedId:
        clients[i].success_prob += 1

    print("* Final aggregation:", selectedId)

    lr: np.float32 = np.median(
        [np.linalg.norm(clients[i].update) for i in selectedId])

    return lr, np.mean([clients[i].momentum for i in selectedId], axis=0)


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
    clients.append(Client(splits[i]['X'], splits[i]['y'], i))

    if i >= client_num - f:
        clients[i].is_byzantine = True

for c in clients:
    c.set_weights(global_weights)

iter = 1
for iter in range(iterations_num):
    print(f"\n\n*** Iteration {iter} ***")
    selectedId: list[int]

    if iter < 10:
        selectedId = [i for i in range(client_num)]
    else:
        selectedId = TS(clients)

    for i in selectedId:
        if not clients[i].is_byzantine:
            print(f"\t + Training on client {clients[i].id}")

            clients[i].train()

    LIE(clients, selectedId)
    lr, global_update = MAB_FL(clients, selectedId, iter)

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
