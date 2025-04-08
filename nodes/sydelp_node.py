import numpy as np

from nodes.node import Node
from utils.typing import Float, FloatArray

class SydelpNode(Node):
    momentum_coeff: Float
    momentum: FloatArray
    contribution_score: int = 0

    def __init__(self,
                 momentum_coeff: Float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_coeff = momentum_coeff
        self.momentum = np.zeros(self.flat_weights.size, dtype='float')

    def update_contribution_score(self, was_selected: bool) -> None:
        reward: int = 1 if was_selected else -1
        self.contribution_score += reward
        # Prevent negative scores
        self.contribution_score = max(0, self.contribution_score)

    def train(self) -> None:
        super().train()

        self.momentum = self.flat_weights + self.momentum_coeff * self.momentum
