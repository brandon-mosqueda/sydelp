from learning.learning import Learning

from typing import TypedDict, Protocol
from utils.utils import ModelParams
from utils.aggregation import fed_avg


class AggFunction(Protocol):
    def __call__(self, models_params: list[ModelParams],
                 *args, **kwargs) -> ModelParams: ...


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
        models: list[ModelParams] = [
            node.get_model_params() for node in self.nodes
        ]
        avg_models: ModelParams = self.aggregation_params['function'](
            models,
            **self.aggregation_params['params']
        )

        self.global_model.set_weights(avg_models)

        for node in self.nodes:
            node.set_model_params(avg_models)

    def model_sharing(self) -> None:
        pass
