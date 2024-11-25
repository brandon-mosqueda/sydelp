import numpy as np

from learning.learning import Learning
from utils.utils import NumArray, flatten_to_original
from utils.aggregation import fed_avg


class FederatedLearning(Learning):
    weights: NumArray
    models_matrix: NumArray

    def __init__(self,
                 weighting_mode: str = "uniform",
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        model_size = sum(w.size for w in self.nodes[0].get_model_params())
        self.models_matrix = np.empty((len(self.nodes), model_size))

        if weighting_mode == "uniform":
            self.weights = (np.repeat(1/len(self.nodes), len(self.nodes))
                            .astype('float'))
        elif weighting_mode == "data":
            self.weights = np.array([node.rows_num for node in self.nodes],
                                    dtype="float")
            self.weights /= self.weights.sum()
        else:
            raise ValueError(f"{weighting_mode} is not a valid weighting mode")

    def iteration_setup(self, iteration_num: int) -> None:
        pass

    def update_models_matrix(self) -> None:
        for i in range(len(self.nodes)):
            self.models_matrix[i] = self.nodes[i].get_flatten_model_params()

    def aggregation(self, iteration_num: int) -> None:
        self.update_models_matrix()

        avg_model: list[NumArray] = flatten_to_original(
            fed_avg(self.models_matrix, self.weights),
            self.global_model
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_params(avg_model)
