import numpy as np

from learning.learning import Learning
from nodes.sydelp_node import SydelpNode
from utils.utils import flat_weights_to_original, set_weights_to_array
from utils.typing import FloatArray
from utils.aggregation import krum


class Sydelp(Learning[SydelpNode]):
    expected_malicious_num: int # beta (krum)
    global_weights: FloatArray

    def __init__(self,
                 expected_malicious_num: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.global_weights = np.empty(self.nodes[0].flat_weights.size,
                                       dtype="float")

        set_weights_to_array(
            self.global_model.get_weights(),
            self.global_weights
        )
        self.expected_malicious_num = expected_malicious_num

    def aggregation(self, iteration_num: int) -> None:
        avg_model: list[FloatArray] = flat_weights_to_original(
            self.global_weights -
            krum(np.array([node.momentum for node in self.nodes],
                          dtype='float'),
                 m=self.expected_malicious_num),
            self.weights_shapes
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_weights(avg_model)
