import numpy as np

from networkx import Graph
from learning.learning import Learning, NodeType, AttackerType
from utils.utils import MyProgressBar, progress_bar, flat_weights_to_original
from utils.aggregation import fed_avg
from utils.typing import Float, IntArray, FloatArray


class Gossip(Learning[NodeType, AttackerType]):
    graph: Graph

    def __init__(self, graph: Graph, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if sorted(list(graph.nodes)) != np.arange(len(self.all_nodes)).tolist():
            raise ValueError("graph.nodes is not equal to nodes")

        self.graph = graph

    def global_model_aggregation(self) -> None:
        # For global aggregation (to compute the metrics) only honest models
        # are considered. Poisoning is prevented (or succeded) locally.
        self.global_flat_weights = fed_avg(
            np.array([node.get_flat_model_weights()
                      for node in self.honest_nodes],
                     dtype='float')
        )

        self.global_model.set_weights(flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        ))

    def aggregation(self, iteration_num: int) -> None:
        for i, node in enumerate(self.all_nodes):
            neighbors: list[int] = list(self.graph.neighbors(i)) + [i]

            node.set_flat_model_weights(fed_avg(
                np.array([self.all_nodes[neigh_i].get_flat_model_weights()
                          for neigh_i in neighbors])
            ))

        self.global_model_aggregation()

    def node_metrics(self, node: NodeType) -> FloatArray:
        predicted: IntArray = node.predict(self.x_testing)

        loss: Float = node.model.evaluate(self.x_testing,
                                          self.y_testing,
                                          verbose=0)

        metrics: list[Float] = [
            self.metrics_params[metric]['function'](
                y_true=self.y_testing,
                y_pred=predicted,
                **self.metrics_params[metric]['params']
            ) for metric in self.metrics_params
        ]
        metrics.append(loss)

        return np.array(metrics)

    def round_metrics(self) -> dict[str, Float]:
        bar: MyProgressBar = progress_bar(len(self.honest_nodes))
        # Loss is added by default on metrics
        values: FloatArray = np.zeros(len(self.metrics_params) + 1,
                                      dtype="float")

        for node in self.honest_nodes:
            bar.next()
            values += self.node_metrics(node)

        bar.finish()
        metrics: list = list(self.metrics_params.keys()) + ['loss']
        values /= len(self.honest_nodes)

        final_metrics: dict[str, Float] = {
            metric: values[i]
            for i, metric in enumerate(metrics)
        }

        global_metrics: dict[str, Float] = super().round_metrics()
        for key, value in global_metrics.items():
            final_metrics['global_' + key] = value

        return final_metrics
