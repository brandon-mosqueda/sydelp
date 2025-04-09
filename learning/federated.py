import numpy as np

from learning.learning import Learning, NodeType, AttackerType
from utils.utils import flat_weights_to_original
from utils.typing import FloatArray
from utils.aggregation import fed_avg


# It is not necessary, but we specify both generic types to have type completion
class FederatedLearning(Learning[NodeType, AttackerType]):
    def aggregation(self, iteration_num: int) -> None:
        self.global_flat_weights = fed_avg(
            np.array([node.get_flat_model_weights()
                      for node in self.all_nodes],
                     dtype='float')
        )
        global_weights: list[FloatArray] = flat_weights_to_original(
            self.global_flat_weights,
            self.weights_shapes
        )

        self.global_model.set_weights(global_weights)

        for node in self.all_nodes:
            node.set_model_weights(global_weights)
