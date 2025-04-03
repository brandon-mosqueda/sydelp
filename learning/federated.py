import numpy as np

from learning.learning import Learning, NodeType
from utils.utils import flat_weights_to_original
from utils.typing import FloatArray
from utils.aggregation import fed_avg


class FederatedLearning(Learning[NodeType]):
    def aggregation(self, iteration_num: int) -> None:
        avg_model: list[FloatArray] = flat_weights_to_original(
            fed_avg(np.array([node.get_flat_model_weights()
                              for node in self.nodes],
                             dtype='float')),
            self.weights_shapes
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_weights(avg_model)
