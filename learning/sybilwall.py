import numpy as np
import pandas as pd

from random import choices
from networkx import Graph
from learning.learning import Learning
from utils.aggregation import fed_avg
from utils.typing import NumArray, Float
from nodes.sybilwall_node import SybilwallNode, HistoricModel
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame, Series
from copy import deepcopy


class Sybilwall(Learning[SybilwallNode]):
    graph: Graph
    distant_propagation_relevance: float # lambda
    confidence: float # kappa

    def __init__(self,
                 graph: Graph,
                 distant_propagation_relevance: float,
                 confidence: float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sorted(list(graph.nodes)) != np.arange(len(self.nodes)).tolist():
            raise ValueError("graph.nodes is not equal to nodes")

        self.graph = graph
        self.distant_propagation_relevance = distant_propagation_relevance
        self.confidence = confidence

    def iteration_setup(self, iteration_num: int) -> None:
        pass

    def gossiping(self) -> None:
        # First share the most recent version of own models with neighborhood
        for node_i, node in enumerate(self.nodes):
            for neighbor_i in self.graph.neighbors(node_i):
                neighbor: SybilwallNode = self.nodes[neighbor_i]

                # Send a deep copy of his model
                send_model: HistoricModel = deepcopy(node.history[node_i])
                send_model['distance'] = 1
                neighbor.replace_in_history(send_model)

        # Now shared a randomly selected model from the historic database.
        # Both sharings are separated to first have the updated models of
        # neighbors.
        for node_i, node in enumerate(self.nodes):
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

                    selected_model: HistoricModel = deepcopy(
                        choices(filtered_hist, weights=probs)[0])
                    selected_model['sender_id'] = node_i
                    selected_model['distance'] += 1
                    neighbor.replace_in_history(selected_model)

    def compute_scores(self,
                       node: SybilwallNode,
                       current_idx: int) -> dict[int, float]:
        similarities: DataFrame = DataFrame(cosine_similarity(
            np.array([hist['model'] for hist in node.history.values()])
        ))
        similarities.columns = list(node.history.keys())
        similarities.set_index(similarities.columns, inplace=True)

        # The current node does not enter in the scoring
        similarities.drop(columns=[current_idx], inplace=True)
        similarities.drop(index=[current_idx], inplace=True)

        # Set the diagonal with lowest similarity
        for i in similarities.columns:
            similarities.loc[i, i] = -1

        max_scores: Series[float] = similarities.max(axis=0) + 1e-5
        final_scores: dict[int, float] = {}

        for i in similarities.index:
            scores_i: list[float] = []
            for j in similarities.columns:
                if i == j:
                    continue

                similarity: float = similarities.loc[i, j] # type: ignore
                # Pardoning
                if max_scores[j] > max_scores[i]:
                    similarity *= max_scores[i] / max_scores[j]

                scores_i.append(similarity)

            final_scores[i] = np.clip(1 - np.max(scores_i), 0, 1)

        max_score: float = np.max(list(final_scores.values()))
        for key in final_scores.keys():
            final_scores[key] /= max_score
            # The log of 0 and negative numbers is not defined and 1 would
            # produce zero division
            final_scores[key] = np.clip(final_scores[key], 0.000001, 0.999999)

            final_scores[key] = (
                self.confidence
                * (np.log(final_scores[key] / (1 - final_scores[key]))
                   + 0.5)
            )

            final_scores[key] = np.clip(final_scores[key], 0, 1)

        return final_scores

    def aggregation(self, iteration_num: int) -> None:
        # Update own local model histories
        for i, node in enumerate(self.nodes):
            model: HistoricModel = {
                'node_id': i,
                'model': node.get_flat_model_weights(),
                'iteration_num': iteration_num,
                'distance': 0,
                'sender_id': i,
            }
            node.add_in_history(model)

        self.gossiping()

        # Aggregation
        for i, node in enumerate(self.nodes):
            scores: dict[int, float] = self.compute_scores(node, i)
            # Own model always has the highest score
            scores[i] = 1

            # Weight normalization (sum of weights is 1)
            total_sum: float = np.sum(list(scores.values()))
            for key in scores.keys():
                scores[key] /= total_sum

            neighbors: list[int] = list(self.graph.neighbors(i))
            neighbors.append(i)

            weights: NumArray = np.array([scores[key] for key in scores.keys()])
            models: NumArray = np.array([
                self.nodes[key].get_flat_model_weights()
                for key in scores.keys()
            ])

            node.set_flat_model_weights(fed_avg(models, weights))

    def round_predictions(self) -> DataFrame:
        all_predictions: list[DataFrame] = []

        for i, node in enumerate(self.nodes):
            if not node.is_malicious:
                preds: DataFrame = node.predict(self.x_testing)
                preds['observed'] = self.y_testing
                preds['node'] = i

                all_predictions.append(preds)

        return pd.concat(all_predictions, ignore_index=True)

    def node_metrics(self, node_preds: DataFrame) -> DataFrame:
        node_i: int = node_preds['node'].iloc[0]

        loss: float = self.nodes[node_i].model.evaluate(self.x_testing,
                                                        self.y_testing,
                                                        verbose=0)

        metrics: dict[str, Float] = {
            metric: self.metrics_params[metric]['function'](
                y_true=node_preds['observed'].to_numpy().astype("int"),
                y_pred=node_preds['predicted'].to_numpy().astype("int"),
                **self.metrics_params[metric]['params']
            ) for metric in self.metrics_params
        }
        metrics['loss'] = loss

        return DataFrame([metrics])

    def round_metrics(self, predictions: DataFrame) -> dict[str, Float]:
        metrics_by_node = (
            predictions.groupby('node')
            .apply(self.node_metrics, include_groups=True)
            .reset_index(level=1, drop=True)
        )

        return dict(metrics_by_node.mean(axis=0)) # type: ignore
