import numpy as np

from learning.learning import Learning
from nodes.sydelp_node import SydelpNode
from utils.utils import flat_weights_to_original
from utils.typing import *
from utils.aggregation import fed_avg, krum_selection


class Sydelp(Learning[SydelpNode]):
    expected_malicious_num: int # beta (krum)
    momentum_coeff: Float
    difficulty_alpha: Float

    def __init__(self,
                 expected_malicious_num: int,
                 momentum_coeff: int,
                 difficulty_alpha: Float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.difficulty_alpha = difficulty_alpha
        self.momentum_coeff = momentum_coeff
        self.expected_malicious_num = expected_malicious_num

    def node_difficulty(self, node: SydelpNode) -> Float:
        phi: int = node.contribution_score

        return (
            (self.iterations_num - self.difficulty_alpha) /
            (self.iterations_num - 1)
        )**phi

    def round_metrics(self) -> dict[str, Float]:
        metrics: dict[str, Float] = super().round_metrics()

        honest_dificulties = np.array([self.node_difficulty(node)
                                       for node in self.nodes
                                       if not node.is_malicious])
        mal_dificulties = np.array([self.node_difficulty(node)
                                    for node in self.nodes
                                    if node.is_malicious])

        metrics['mean_honest_difficulty'] = honest_dificulties.mean()
        metrics['sd_honest_difficulty'] = honest_dificulties.std()

        metrics['mean_mal_difficulty'] = mal_dificulties.mean()
        metrics['sd_mal_difficulty'] = mal_dificulties.std()

        return metrics

    def aggregation(self, iteration_num: int) -> None:
        selected_idx: BoolArray = krum_selection(
            np.array([node.momentum
                      for node in self.nodes],
                     dtype='float'),
            m=self.expected_malicious_num
        )

        for (i, was_selected) in enumerate(selected_idx):
            self.nodes[i].update_contribution_score(was_selected)

        self.global_flat_weights = fed_avg(
            np.array([node.momentum
                      for (i, node) in enumerate(self.nodes)
                      if selected_idx[i]],
                     dtype='float')
        )

        global_weights: list[FloatArray] = flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(global_weights)

        for node in self.nodes:
            node.set_model_weights(global_weights)
