import numpy as np

from typing import Callable
from pandas import DataFrame, concat
from time import time
from numpy.typing import NDArray
from utils.utils import NNumeric, Weights, elapsed_time
from learning.collaborative import CollaborativeLearning
from utils.aggregation import fed_avg

AggFunct = Callable[[list[Weights]], Weights]

class FederatedLearning(CollaborativeLearning):
    aggregation_function: AggFunct
    aggregation_params: dict

    def __init__(self,
                 aggregation_function: AggFunct = fed_avg,
                 aggregation_params: dict = {},
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.aggregation_function = aggregation_function
        self.aggregation_params = aggregation_params

    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = self.aggregation_function(
            weights,
            **self.aggregation_params
        )

        self.global_model.set_weights(avg_weights)

        for node in self.nodes:
            node.set_weights(avg_weights)

    def round_metrics(self, round: int, start_time: float) -> dict:
        accuracy: NDArray[NNumeric]
        loss, accuracy = self.global_model.evaluate(self.x_testing,
                                                    self.y_testing,
                                                    verbose=0)

        metrics = {
            'round': round,
            'time': elapsed_time(start_time, time()),
            'accuracy': accuracy,
            'loss': loss
        }

        return metrics

    def round_predictions(self, round: int) -> DataFrame:
        probs: DataFrame = DataFrame(self.global_model.predict(
            self.x_testing,
            verbose=0
        ))

        data: DataFrame = DataFrame({
            "round": round,
            "observed": self.y_testing,
            "predicted": np.argmax(probs, axis=1)
        })

        return concat([data, probs], axis=1)
