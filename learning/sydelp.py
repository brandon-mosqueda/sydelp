import numpy as np

from learning.learning import Learning
from utils.utils import NumArray, set_flatten_weights
from utils.aggregation import krum


class Sydelp(Learning):
    weighting_mode: str
    expected_malicious_num: int
    data_sizes: NumArray

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

    def aggregation(self) -> None:
        all_models: NumArray = np.array(
            [node.get_model_params() for node in self.nodes])

        avg_model: NumArray = krum(all_models,
                                   m=self.expected_malicious_num,
                                   weighting_mode=self.weighting_mode,
                                   data_sizes=self.data_sizes)
        set_flatten_weights(self.global_model, avg_model)

        for node in self.nodes:
            node.set_model_params(avg_model)

    def model_sharing(self) -> None:
        pass
