import numpy as np

from learning.learning import Learning
from typing import TypedDict, Protocol
from utils.utils import NumArray, set_flatten_weights
from utils.aggregation import fed_avg


class AggFunction(Protocol):
    def __call__(self, models_params: NumArray,
                 *args, **kwargs) -> NumArray: ...


class AggParams(TypedDict):
    function: AggFunction
    params: dict


class FederatedLearning(Learning):
    aggregation_params: AggParams

    def __init__(self,
                 aggregation_params: AggParams = {
                     'function': fed_avg,
                     'params': {}
                 },
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aggregation_params = aggregation_params

    def aggregation(self) -> None:
        models: NumArray = np.array([
            node.get_model_params() for node in self.nodes])
        avg_models: NumArray = self.aggregation_params['function'](
            models,
            **self.aggregation_params['params']
        )

        set_flatten_weights(self.global_model, avg_models)

        for node in self.nodes:
            node.set_model_params(avg_models)

    def model_sharing(self) -> None:
        pass
