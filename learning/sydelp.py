import numpy as np

from learning.learning import Learning
from nodes.sydelp_node import SydelpNode
from utils.utils import flat_weights_to_original
from utils.typing import FloatArray
from utils.aggregation import krum


class Sydelp(Learning[SydelpNode]):
    expected_malicious_num: int # beta (krum)

    def __init__(self, expected_malicious_num: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.expected_malicious_num = expected_malicious_num

    def aggregation(self, iteration_num: int) -> None:
        self.global_flat_weights = krum(
            np.array([node.get_flat_model_weights()
                      for node in self.nodes],
                     dtype='float'),
            m=self.expected_malicious_num
        )

        avg_model: list[FloatArray] = flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_weights(avg_model)
