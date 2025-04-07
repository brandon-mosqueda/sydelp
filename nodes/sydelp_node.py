import numpy as np

from utils.typing import FloatArray, Float, IntArray, NumArray
from nodes.node import Node
from utils.utils import compute_gradient, set_weights_to_array


class SydelpNode(Node):
    momentum: FloatArray
    momentum_coefficient: Float

    def __init__(self, momentum_coefficient: Float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_coefficient = momentum_coefficient
        self.momentum = np.zeros(self.flat_weights.size, dtype="float")


    def train(self) -> None:
        idx: IntArray = np.random.choice(
            np.arange(self.x.shape[0]),
            size=min(self.x.shape[0], self.batch_size),
            replace=False
        )

        x: NumArray = self.x[idx]
        y: IntArray = self.y[idx]

        set_weights_to_array(
            compute_gradient(self.model, x, y),
            self.flat_weights
        )

        self.momentum = (
            (self.momentum_coefficient * self.momentum) +
            ((1 - self.momentum_coefficient) * self.flat_weights)
        )
