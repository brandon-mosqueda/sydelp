import numpy as np

from learning.learning import Learning
from nodes.sydelp_node import SydelpNode
from utils.utils import flat_weights_to_original, MyProgressBar, progress_bar
from utils.typing import *
from utils.aggregation import fed_avg, krum_selection


class Sydelp(Learning[SydelpNode]):
    expected_malicious_num: int # beta (krum)
    momentum_coeff: Float
    difficulty_alpha: Float
    models_per_iteration: int
    considered_idx: IntArray

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

        if mal_dificulties:
            metrics['mean_mal_difficulty'] = mal_dificulties.mean()
            metrics['sd_mal_difficulty'] = mal_dificulties.std()
        else:
            metrics['mean_mal_difficulty'] = np.nan
            metrics['sd_mal_difficulty'] = np.nan

        return metrics

    def training(self) -> None:
        malicious_idx: IntArray = np.array([
            i
            for (i, node) in enumerate(self.nodes)
            if node.is_malicious
        ])
        honest_idx: IntArray = np.array([
            i
            for (i, node) in enumerate(self.nodes)
            if not node.is_malicious
        ])
        honest_idx = np.random.choice(
            honest_idx,
            size=self.models_per_iteration - malicious_idx.size,
            replace=False
        )

        # The malicious models are always taken because we assume that the
        # attacker can always finish the train for them before the honest nodes
        self.considered_idx = np.concatenate([malicious_idx, honest_idx])

        bar: MyProgressBar = progress_bar(honest_idx.size)

        for i in honest_idx:
            self.nodes[i].train()
            bar.next()

        bar.finish()

    def aggregation(self, iteration_num: int) -> None:
        were_selected: BoolArray = krum_selection(
            np.array([node.momentum
                      for (i, node) in enumerate(self.nodes)
                      if i in self.considered_idx],
                     dtype='float'),
            m=self.expected_malicious_num
        )

        for (was_selected, idx) in zip(were_selected, self.considered_idx):
            self.nodes[idx].update_contribution_score(was_selected)

        self.global_flat_weights = fed_avg(
            np.array([self.nodes[idx].momentum
                      for (was_selected, idx) in zip(were_selected,
                                                     self.considered_idx)
                      if was_selected],
                     dtype='float')
        )

        global_weights: list[FloatArray] = flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(global_weights)

        for node in self.nodes:
            node.set_model_weights(global_weights)
