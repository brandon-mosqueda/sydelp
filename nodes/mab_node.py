import numpy as np

from utils.typing import NumArray, Int
from nodes.node import Node


class MabNode(Node):
    success_prob: float
    fail_prob: float
    momentum: NumArray
    update: NumArray
    seleted_epoch: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.success_prob = 1
        self.fail_prob = 1
        self.seleted_epoch = 0

        self.momentum = np.zeros(self.flat_weights.size)
        self.update = np.zeros(self.flat_weights.size)

    def set_update(self, global_weights: NumArray) -> None:
        self.update = self.flat_weights - global_weights
