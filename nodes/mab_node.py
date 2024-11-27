import numpy as np

from utils.utils import get_flatten_weights
from utils.typing import NumArray
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

        total_size: int = sum(w.size for w in self.model.get_weights())
        self.momentum = np.zeros(total_size)
        self.update = np.zeros(total_size)

    def set_update(self, global_weights: NumArray) -> None:
        self.update = get_flatten_weights(self.model) - global_weights
