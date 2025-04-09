import numpy as np
import utils.utils as utils

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from networkx import Graph, connected_components
from utils.utils import MyProgressBar
from utils.typing import IntArray, Float, FloatArray
from utils.metrics import cos_similarity
from learning.learning import Learning, AttackerType
from nodes.mab_node import MabNode


class Mab(Learning[MabNode, AttackerType]):
    selected_idx: IntArray
    warm_up_iterations: int
    alpha: float
    miu: float
    c_max: float
    c_min: float
    pca_components: float

    def __init__(self,
                 warm_up_iterations: int = 10,
                 alpha: float = 0,
                 miu: float = 0.1,
                 c_max: float = 0.3,
                 c_min: float = 0.1,
                 pca_components: float = 0.95,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warm_up_iterations = warm_up_iterations
        self.alpha = alpha
        self.miu = miu
        self.c_max = c_max
        self.c_min = c_min
        self.pca_components = pca_components

    def client_beta_selection(self) -> IntArray:
        selected_ids: list[int] = []

        for i, node in enumerate(self.all_nodes):
            p: float = np.random.beta(node.success_prob, node.fail_prob)

            if p >= 0.9 or (p > 0.2 and np.random.random() < p):
                selected_ids.append(i)

        if len(selected_ids) < 2:
            selected_ids = [i for i in range(len(self.all_nodes))]

        return np.array(selected_ids)

    def iteration_setup(self, iteration_num: int) -> None:
        if iteration_num < self.warm_up_iterations:
            self.selected_idx = np.arange(len(self.all_nodes))
        else:
            self.selected_idx = self.client_beta_selection()

    def training(self) -> None:
        bar: MyProgressBar = utils.progress_bar(len([
            i
            for i in self.selected_idx
            if not self.all_nodes[i].is_malicious
        ]))

        for i in self.selected_idx:
            if not self.all_nodes[i].is_malicious:
                self.all_nodes[i].train()
                bar.next()

        bar.finish()

    # Remove possible sybil ids selected_idx
    def remove_sybils_idx(self, iteration_num: int) -> None:
        sybil_similarity_threshold: float = max(
            self.c_max * np.exp(-iteration_num / 20),
            self.c_min
        )

        graph: Graph = Graph()
        edges: list[tuple] = []

        for i in range(len(self.selected_idx)):
            for j in range(i + 1, len(self.selected_idx)):
                similarity: Float = cos_similarity(
                    self.all_nodes[self.selected_idx[i]].momentum,
                    self.all_nodes[self.selected_idx[j]].momentum
                )

                if similarity > sybil_similarity_threshold:
                    # We add the indices of the elements relative to
                    # selected_idx so we can extract this indices from
                    # max_connected_component and then remove them of
                    # selected_idx
                    edges.append((i, j))

        graph.add_nodes_from(np.arange(len(self.selected_idx)))
        graph.add_edges_from(edges)

        # max connected component
        remove_idx: list[int] = list(sorted(
            connected_components(graph),
            key=len,
            reverse=True
        )[0])

        # If almost all selected indices are going to be deleted, keep them all
        # We need at least two models to avoid errors in cluster operations
        if len(self.selected_idx) - len(remove_idx) >= 2:
            for i in self.selected_idx[remove_idx]:
                self.all_nodes[i].fail_prob += 1

            self.selected_idx = np.delete(self.selected_idx, remove_idx)

    def aggregation(self, iteration_num: int) -> None:
        # self.selected_idx is update before training in iteration_setup
        for i in self.selected_idx:
            node: MabNode = self.all_nodes[i]
            node.set_update(self.global_flat_weights)

            node.momentum = (
                node.update
                + (self.miu**(iteration_num - node.selected_epoch)
                   * node.momentum)
            )

            node.momentum = node.momentum / np.linalg.norm(node.momentum)
            node.selected_epoch = iteration_num

        # Remove sybils
        self.remove_sybils_idx(iteration_num)

        selected_updates: FloatArray = np.array(
            [self.all_nodes[i].momentum for i in self.selected_idx])

        pca: PCA = PCA(n_components=self.pca_components)
        X_reduced = pca.fit_transform(selected_updates)

        estimator: AgglomerativeClustering = AgglomerativeClustering(2)
        estimator.fit(X_reduced)
        cluster_labels: IntArray = estimator.labels_

        # cluster_labels is the same length as self.selected_idx, but we want to
        # use these ids from the cluster to index self.all_nodes
        ids_cluster1: IntArray = self.selected_idx[
            np.where(cluster_labels == 0)[0]
        ]
        ids_cluster2: IntArray = self.selected_idx[
            np.where(cluster_labels == 1)[0]
        ]

        mean_cluster1: FloatArray = np.mean(
            [self.all_nodes[i].momentum for i in ids_cluster1],
            axis=0
        )
        mean_cluster2: FloatArray = np.mean(
            [self.all_nodes[i].momentum for i in ids_cluster2],
            axis=0
        )

        clusters_similarity: Float = cos_similarity(mean_cluster1,
                                                    mean_cluster2)

        if clusters_similarity < self.alpha:
            smallest_cluster, largest_cluster = (
                (ids_cluster2, ids_cluster1)
                if len(ids_cluster1) > len(ids_cluster2)
                else (ids_cluster1, ids_cluster2)
            )

            for i in smallest_cluster:
                self.all_nodes[i].fail_prob += 1

            self.selected_idx = largest_cluster

        # Increase the prob for the remaining nodes (largest cluster)
        for i in self.selected_idx:
            self.all_nodes[i].success_prob += 1

        lr: np.float_ = np.median([np.linalg.norm(self.all_nodes[i].update)
                                   for i in self.selected_idx])

        global_update: FloatArray = np.mean(
            [self.all_nodes[i].momentum for i in self.selected_idx],
            axis=0
        )

        self.global_flat_weights = self.global_flat_weights + lr * global_update

        global_weights: list[FloatArray] = utils.flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(global_weights)

        for node in self.all_nodes:
            node.set_model_weights(global_weights)
