import numpy as np

from learning.learning import Learning
from utils.utils import NumArray, flatten_to_original
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
        model_size = sum(w.size for w in self.nodes[0].get_model_params())
        self.models_matrix = np.empty((len(self.nodes), model_size))

    def iteration_setup(self, iteration_num: int) -> None:
        pass

    def update_models_matrix(self) -> None:
        for i in range(len(self.nodes)):
            self.models_matrix[i] = self.nodes[i].get_flatten_model_params()

    def aggregation(self, iteration_num: int) -> None:
        self.update_models_matrix()

        avg_model: list[NumArray] = flatten_to_original(
            krum(self.models_matrix,
                 m=self.expected_malicious_num,
                 weighting_mode=self.weighting_mode,
                 data_sizes=self.data_sizes),
            self.global_model
        )

        self.global_model.set_weights(avg_model)

        for node in self.nodes:
            node.set_model_params(avg_model)
