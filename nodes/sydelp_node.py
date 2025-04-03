import numpy as np

from utils.typing import FloatArray, Float
from nodes.node import Node


class SydelpNode(Node):
    momentum: FloatArray
    momentum_coefficient: Float

    def __init__(self, momentum_coefficient: Float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_coefficient = momentum_coefficient
        self.momentum = np.zeros(self.flat_weights.size, dtype="float")

    def train(self) -> None:
        super().train()

        self.momentum = (self.momentum_coefficient * self.momentum +
                         (1 - self.momentum_coefficient) * self.flat_weights)
