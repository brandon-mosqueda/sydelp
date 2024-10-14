from numpy.typing import NDArray
from utils import NNumeric, Weights
from collaborative_learning import CollaborativeLearning
from aggregation import fed_avg


class FederatedLearning(CollaborativeLearning):
    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = fed_avg(weights)

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
