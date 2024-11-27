import numpy as np

from learning.learning import Learning
from utils.utils import flat_weights_to_original
from utils.typing import NumArray
from utils.aggregation import fed_avg


class FederatedLearning(Learning):
    weights: NumArray
    models_matrix: NumArray

    def __init__(self,
                 weighting_mode: str = "uniform",
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.models_matrix = np.empty((len(self.nodes),
                                       self.nodes[0].flat_weights.size),
                                      dtype="float32")

        if weighting_mode == "uniform":
            self.weights = (np.repeat(1/len(self.nodes), len(self.nodes))
                            .astype('float'))
        elif weighting_mode == "data":
            self.weights = np.array([node.rows_num for node in self.nodes],
                                    dtype="float")
            self.weights /= self.weights.sum()
        else:
            raise ValueError(f"{weighting_mode} is not a valid weighting mode")

    def update_models_matrix(self) -> None:
        for i, node in enumerate(self.nodes):
            self.models_matrix[i] = node.get_flat_model_weights()

    def aggregation(self, iteration_num: int) -> None:
        self.update_models_matrix()

        avg_model: list[NumArray] = flat_weights_to_original(
            fed_avg(self.models_matrix, self.weights),
            self.weights_shapes
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_weights(avg_model)
