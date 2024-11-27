import numpy as np

from learning.learning import Learning
from utils.utils import flat_weights_to_original
from utils.typing import NumArray
from utils.aggregation import krum


class Sydelp(Learning):
    weighting_mode: str
    expected_malicious_num: int
    data_sizes: NumArray
    models_matrix: NumArray

    def __init__(self,
                 expected_malicious_num: int,
                 weighting_mode: str = "uniform",
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if weighting_mode not in ["uniform", "data", "score"]:
            raise ValueError(f"{weighting_mode} is not a valid weighting mode")

        self.weighting_mode = weighting_mode
        self.expected_malicious_num = expected_malicious_num
        self.data_sizes = np.array([node.rows_num for node in self.nodes])
        self.models_matrix = np.empty((len(self.nodes),
                                       self.nodes[0].flat_weights.size),
                                      dtype="float32")

    def update_models_matrix(self) -> None:
        for i, node in enumerate(self.nodes):
            self.models_matrix[i] = node.get_flat_model_weights()

    def aggregation(self, iteration_num: int) -> None:
        self.update_models_matrix()

        avg_model: list[NumArray] = flat_weights_to_original(
            krum(self.models_matrix,
                 m=self.expected_malicious_num,
                 weighting_mode=self.weighting_mode,
                 data_sizes=self.data_sizes),
            self.weights_shapes
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_weights(avg_model)
