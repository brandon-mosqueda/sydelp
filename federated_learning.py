import numpy as np

from pandas import DataFrame
from numpy.typing import NDArray
from utils import NNumeric, Weights
from collaborative_learning import CollaborativeLearning

class FederatedLearning(CollaborativeLearning):
    @staticmethod
    def fed_avg(weights: list[Weights]) -> Weights:
        avg_weights: Weights = []

        for i in range(len(weights[0])):
            avg_weights.append(np.mean([w[i] for w in weights], axis=0))

        return avg_weights

    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = self.fed_avg(weights)

        self.global_model.set_weights(avg_weights)

        for node in self.nodes:
            node.set_weights(avg_weights)

    def evaluate(self) -> dict:
        accuracy: NDArray[NNumeric]
        _, accuracy = self.global_model.evaluate(self.x_testing,
                                                 self.y_testing,
                                                 verbose=0)
        metrics = {'accuracy': accuracy}

        return metrics
