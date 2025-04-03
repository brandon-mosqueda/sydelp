import numpy as np

from networkx import Graph
from learning.learning import Learning, NodeType
from utils.utils import MyProgressBar, progress_bar, flat_weights_to_original
from utils.aggregation import fed_avg
from utils.typing import Float, IntArray, FloatArray


class Gossip(Learning[NodeType]):
    graph: Graph

    def __init__(self, graph: Graph, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sorted(list(graph.nodes)) != np.arange(len(self.nodes)).tolist():
            raise ValueError("graph.nodes is not equal to nodes")

        self.graph = graph

    def global_model_aggregation(self) -> None:
        # For global aggregation (to compute the metrics) only honest models
        # are considered. Poisoning is prevented (or succeded) locally.
        self.global_model.set_weights(flat_weights_to_original(
            fed_avg(np.array([node.get_flat_model_weights()
                              for node in self.nodes
                              if not node.is_malicious])),
            self.weights_shapes
        ))

    def aggregation(self, iteration_num: int) -> None:
        for i, node in enumerate(self.nodes):
            neighbors: list[int] = list(self.graph.neighbors(i)) + [i]

            node.set_flat_model_weights(fed_avg(
                np.array([self.nodes[neigh_i].get_flat_model_weights()
                          for neigh_i in neighbors])
            ))

        self.global_model_aggregation()

    def node_metrics(self,
                     observed: IntArray,
                     predicted: IntArray,
                     node_i: int) -> FloatArray:
        loss: Float = self.nodes[node_i].model.evaluate(self.x_testing,
                                                        self.y_testing,
                                                        verbose=0)

        metrics: list[Float] = [
            self.metrics_params[metric]['function'](
                y_true=observed,
                y_pred=predicted,
                **self.metrics_params[metric]['params']
            ) for metric in self.metrics_params
        ]
        metrics.append(loss)

        return np.array(metrics)

    def round_metrics(self) -> dict[str, Float]:
        bar: MyProgressBar = progress_bar(len(self.nodes))
        # Loss is added by default on metrics
        values: FloatArray = np.zeros(len(self.metrics_params) + 1,
                                      dtype="float")

        for i, node in enumerate(self.nodes):
            bar.next()

            if not node.is_malicious:
                predicted: IntArray = node.predict(self.x_testing)
                values += self.node_metrics(self.y_testing, predicted, i)

        bar.finish()
        metrics: list = list(self.metrics_params.keys()) + ['loss']
        values /= sum([not node.is_malicious for node in self.nodes])

        final_metrics: dict[str, Float] = {
            metric: values[i]
            for i, metric in enumerate(metrics)
        }

        global_metrics: dict[str, Float] = super().round_metrics()
        for key, value in global_metrics.items():
            final_metrics['global_' + key] = value

        return final_metrics
