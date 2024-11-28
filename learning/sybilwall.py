import numpy as np

from random import choices
from learning.gossip import Gossip
from utils.utils import MyProgressBar, progress_bar
from utils.aggregation import fed_avg
from utils.typing import Float, FloatArray
from nodes.sybilwall_node import SybilwallNode, HistoricModel
from sklearn.metrics.pairwise import cosine_similarity


class Sybilwall(Gossip[SybilwallNode]):
    distant_propagation_relevance: float # lambda
    confidence: float # kappa

    def __init__(self,
                 distant_propagation_relevance: float,
                 confidence: float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.distant_propagation_relevance = distant_propagation_relevance
        self.confidence = confidence

    def gossiping(self) -> None:
        print("\t+ Gossiping")
        bar: MyProgressBar = progress_bar(len(self.nodes) * 2)

        # First share the most recent version of own models with neighborhood
        for node_i, node in enumerate(self.nodes):
            bar.next()
            node_model: HistoricModel = node.history[node_i]
            node_model['distance'] = 1

            for neighbor_i in self.graph.neighbors(node_i):
                self.nodes[neighbor_i].replace_in_history(**node_model)

            node_model['distance'] = 0

        # Then shared a randomly selected model from the historic database.
        # Both sharings are separated to first have the updated models of
        # neighbors before the random selection.
        for node_i, node in enumerate(self.nodes):
            bar.next()
            for neighbor_i in self.graph.neighbors(node_i):
                neighbor: SybilwallNode = self.nodes[neighbor_i]

                # Remove own model (already sent) and models coming from the
                # neighbor
                filtered_hist: list[HistoricModel] = [
                    hist
                    for hist in node.history.values()
                    if hist['node_id'] not in [node_i, neighbor_i]
                        and hist['sender_id'] != neighbor_i
                ]

                if filtered_hist:
                    probs = [
                        # lamb * e^{-lamb * d}
                        self.distant_propagation_relevance
                            * np.exp(-self.distant_propagation_relevance
                                     * hist['distance'])
                        for hist in filtered_hist
                    ]

                    selected_model: HistoricModel = choices(filtered_hist,
                                                            weights=probs)[0]
                    prev_sender: int = selected_model['sender_id']
                    selected_model['sender_id'] = node_i
                    selected_model['distance'] += 1
                    neighbor.replace_in_history(**selected_model)
                    selected_model['distance'] -= 1
                    selected_model['sender_id'] = prev_sender

        bar.finish()

    def compute_scores(self,
                       node: SybilwallNode,
                       current_idx: int) -> dict[int, Float]:
        similarities: FloatArray = cosine_similarity(
            np.array([hist['model']
                      for hist in node.history.values()
                      if hist['node_id'] != current_idx])
        )
        idx: list[int] = list(node.history.keys())
        idx.remove(current_idx)
        n: int = len(idx)

        # Set the diagonal with lowest similarity
        np.fill_diagonal(similarities, -1)

        # Apply pardoning
        max_simils: FloatArray = np.max(similarities, axis=1)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if max_simils[i] < max_simils[j]:
                    similarities[i, j] *= max_simils[i] / max_simils[j]

        # Compute new maximum weights
        weights: FloatArray = 1 - similarities.max(axis=1)
        weights /= weights.max()
        # Prevent zero division
        weights[weights == 1] = 0.999999
        # Log is not defined for numbers < 0
        weights[weights < 0] = 0.000001
        weights[:] = self.confidence * (np.log(weights / (1 - weights)) + 0.5)
        np.clip(weights, 0, 1, out=weights)

        # Add 1 for the current index that always have the maximum score
        total: Float = weights.sum() + 1
        weights /= total
        scores: dict[int, Float] = {idx[i]: weights[i] for i in range(n)}
        scores[current_idx] = 1 / total

        return scores

    def aggregation(self, iteration_num: int) -> None:
        # Update own local model histories
        for i, node in enumerate(self.nodes):
            node.add_in_history(
                node_id=i,
                model=node.get_flat_model_weights(),
                iteration_num=iteration_num,
                distance=0,
                sender_id=i,
            )

        self.gossiping()

        print("\t+ Aggregating")
        bar: MyProgressBar = progress_bar(len(self.nodes) * 2)

        # Aggregation
        for i, node in enumerate(self.nodes):
            bar.next()
            scores: dict[int, Float] = self.compute_scores(node, i)
            neighbors: list[int] = list(self.graph.neighbors(i)) + [i]

            weights: FloatArray = np.array([scores[neigh_i]
                                            for neigh_i in neighbors])

            models: FloatArray = np.array([
                self.nodes[neigh_i].get_flat_model_weights()
                for neigh_i in neighbors
            ])

            node.set_flat_model_weights(fed_avg(models, weights))

        bar.finish()

        self.global_model_aggregation()
