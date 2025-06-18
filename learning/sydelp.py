import numpy as np

from learning.learning import Learning
from nodes.sydelp_node import SydelpNode
from attack.sydelp_attacker import SydelpAttacker
from utils.utils import flat_weights_to_original, MyProgressBar, progress_bar
from utils.typing import *
from utils.aggregation import fed_avg, krum_selection


class Sydelp(Learning[SydelpNode, SydelpAttacker]):
    expected_malicious_num: int  # beta (krum)
    momentum_coeff: Float
    difficulty_alpha: Float
    models_per_iteration: int
    # Each iteration this list is updated with models_per_iteration nodes
    aggregation_nodes: list[SydelpNode]

    def __init__(self,
                 expected_malicious_num: int,
                 momentum_coeff: int,
                 difficulty_alpha: Float,
                 models_per_iteration: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.models_per_iteration = models_per_iteration
        self.difficulty_alpha = difficulty_alpha
        self.momentum_coeff = momentum_coeff
        self.expected_malicious_num = expected_malicious_num

    def round_metrics(self) -> dict[str, Float]:
        metrics: dict[str, Float] = super().round_metrics()

        metrics['malicious_num'] = len(self.mal_nodes)
        honest_dificulties: FloatArray = np.array([
            node.compute_difficulty()
            for node in self.honest_nodes
        ])
        mal_dificulties: FloatArray = np.array([
            node.compute_difficulty()
            for node in self.mal_nodes
        ])

        metrics['mean_honest_difficulty'] = honest_dificulties.mean()
        metrics['sd_honest_difficulty'] = honest_dificulties.std()

        if mal_dificulties.size > 0:
            metrics['mal_total_difficulty'] = mal_dificulties.sum()
            metrics['mean_mal_difficulty'] = mal_dificulties.mean()
            metrics['sd_mal_difficulty'] = mal_dificulties.std()
        else:
            metrics['mal_total_difficulty'] = np.nan
            metrics['mean_mal_difficulty'] = np.nan
            metrics['sd_mal_difficulty'] = np.nan

        return metrics

    def training(self) -> None:
        # The malicious models are always taken because we assume that the
        # attacker can always finish the train for them before the honest nodes
        # (limited to his computing power). To simulate that honest nodes finish
        # after and as they have similar computing capabilities, we take the
        # honest nodes with lowest difficulties. In case of a tie, we randomly
        # choose them.
        honest_difficulties: FloatArray = np.array(
            [node.compute_difficulty() for node in self.honest_nodes],
            dtype="float"
        )
        ordered_indices: IntArray = np.argsort(honest_difficulties)

        if np.any(honest_difficulties == 1):
            # In the case where there are several nodes with maximum difficulty,
            # they will be taken at random.
            first_one_index = np.argmax(
                honest_difficulties[ordered_indices] == 1)
            ordered_indices[first_one_index:] = np.random.permutation(
                ordered_indices[first_one_index:])

        honest_agg_idx: IntArray = ordered_indices[
            0:(self.models_per_iteration - len(self.mal_nodes))]
        honest_agg_nodes: list[SydelpNode] = [
            node
            for i, node in enumerate(self.honest_nodes)
            if i in honest_agg_idx
        ]

        self.aggregation_nodes = self.mal_nodes + honest_agg_nodes

        bar: MyProgressBar = progress_bar(len(honest_agg_nodes))

        for node in honest_agg_nodes:
            node.train()
            bar.next()

        bar.finish()

    def aggregation(self, iteration_num: int) -> None:
        were_selected: BoolArray = krum_selection(
            np.array([node.momentum
                      for node in self.aggregation_nodes],
                     dtype='float'),
            m=self.expected_malicious_num
        )

        for was_selected, node in zip(were_selected, self.aggregation_nodes):
            node.update_contribution_score(was_selected)

        self.global_flat_weights = fed_avg(
            np.array([node.momentum
                      for was_selected, node in zip(were_selected,
                                                    self.aggregation_nodes)
                      if was_selected],
                     dtype='float')
        )

        global_weights: list[FloatArray] = flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(global_weights)

        for node in self.honest_nodes:
            node.set_model_weights(global_weights)

        for node in self.mal_nodes:
            node.set_model_weights(global_weights)

        # After the end of the iteration (after aggregation), we allow new nodes
        # to join. This is put here because it is the only system where this
        # dynamism is allowed. self.mal_nodes is automatically updated as it is
        # a reference list to self.attacker.nodes
        if self.attacker is not None:
            self.attacker.update_nodes()
