import numpy as np

from learning.learning import Learning
from utils.utils import NumArray, flatten_to_original
from utils.aggregation import fed_avg


class FederatedLearning(Learning):
    weights: NumArray

    def __init__(self,
                 weighting_mode: str = "uniform",
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if weighting_mode == "uniform":
            self.weights = (np.repeat(1/len(self.nodes), len(self.nodes))
                            .astype('float'))
        elif weighting_mode == "data":
            self.weights = np.array([node.rows_num for node in self.nodes],
                                    dtype="float")
            self.weights /= self.weights.sum()
        else:
            raise ValueError(f"{weighting_mode} is not a valid weighting mode")

    def aggregation(self) -> None:
        avg_model: list[NumArray] = flatten_to_original(
            fed_avg(self.models_matrix, self.weights),
            self.global_model
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_params(avg_model)

    def model_sharing(self) -> None:
        pass
