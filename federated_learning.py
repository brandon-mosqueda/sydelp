import numpy as np

from pandas import DataFrame, concat
from time import time
from numpy.typing import NDArray
from utils import NNumeric, Weights, elapsed_time
from collaborative_learning import CollaborativeLearning
from aggregation import fed_avg


class FederatedLearning(CollaborativeLearning):
    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = fed_avg(weights)

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
