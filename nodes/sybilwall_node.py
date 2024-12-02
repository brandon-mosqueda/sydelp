from __future__ import annotations
import numpy as np

from random import choices
from typing import TypedDict, Union
from utils.typing import FloatArray, Float
from nodes.node import Node
from utils.aggregation import fed_avg
from sklearn.metrics.pairwise import cosine_similarity


class HistoricModel(TypedDict):
    node_id: int       # p
    model: FloatArray  # h
    iteration_num: int # r
    distance: int      # d
    sender_id: int     # f


class SybilwallNode(Node):
    id: int
    confidence: float  # kappa
    distant_propagation_relevance: float  # lambda
    history: dict[int, HistoricModel]
    own_history_weights: FloatArray
    neighbors: dict[int, SybilwallNode]

    def __init__(self,
                 confidence: float,
                 distant_propagation_relevance: float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.confidence = confidence
        self.distant_propagation_relevance = distant_propagation_relevance
        self.history = {}
        self.own_history_weights = np.zeros(self.flat_weights.size,
                                            dtype="float")

    # model should be a copy, not a reference
    def update_own_history(self) -> None:
        self.own_history_weights += self.flat_weights

    # model should be a copy, not a reference
    def replace_in_history(self,
                           node_id: int,
                           model: FloatArray,
                           iteration_num: int,
                           distance: int,
                           sender_id: int):
        # Own history is managed through update_own_history function
        if node_id == self.id:
            return

        hist: Union[None, HistoricModel] = self.history.get(node_id, None)

        if hist is None:
            self.history[node_id] = {
                'node_id': node_id,
                'model': model.copy(),
                'iteration_num': iteration_num,
                'distance': distance,
                'sender_id': sender_id,
            }
        elif iteration_num > hist['iteration_num']:
            hist['model'][:] = model
            hist['iteration_num'] = iteration_num
            hist['distance'] = distance
            hist['sender_id'] = sender_id

    def gossip(self) -> None:
        for neighbor_i, neighbor in self.neighbors.items():
            # Remove models originated or coming from neighbor, this
            # prevents to include in their histories information about
            # themselves
            filtered_hist: list[HistoricModel] = [
                hist
                for hist in self.history.values()
                if hist['node_id'] != neighbor_i
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
                selected_model['sender_id'] = self.id
                selected_model['distance'] += 1
                neighbor.replace_in_history(**selected_model)
                selected_model['distance'] -= 1
                selected_model['sender_id'] = prev_sender


    def compute_scores(self) -> dict[int, Float]:
        similarities: FloatArray = cosine_similarity(
            np.array([hist['model'] for hist in self.history.values()])
        )
        idx: list[int] = list(self.history.keys())
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
        weights[weights <= 0] = 0.000001
        weights[:] = self.confidence * (np.log(weights / (1 - weights)) + 0.5)
        np.clip(weights, 0, 1, out=weights)

        return {idx[i]: weights[i] for i in range(n)}

    def aggregate(self):
        scores: dict[int, Float] = self.compute_scores()
        neighbors_idx: list[int] = list(self.neighbors.keys())

        # Own score has always the highest value
        weights: FloatArray = np.array([scores[neigh_i]
                                        for neigh_i in neighbors_idx]
                                       + [1])
        # Normalize the final weights
        weights /= weights.sum()

        models: FloatArray = np.array([
            self.neighbors[neigh_i].get_flat_model_weights()
            for neigh_i in neighbors_idx
        ] + [self.get_flat_model_weights()])

        self.set_flat_model_weights(fed_avg(models, weights))
