import numpy as np

from nodes.node import Node
from nodes.malicious_node import MaliciousNode
from numpy.typing import NDArray


# When attacking the model is normally trained but using the flipped labels
class TargetedLabelFlippingNode(MaliciousNode):
    source: int
    target: int
    source_indices: NDArray[np.int64]

    def __init__(self,
                 source: int,
                 target: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.source = source
        self.target = target
        self.source_indices = np.where(self.y == self.source)[0]

    def attack(self) -> None:
        self.y[self.source_indices] = self.target

        # This prevents infinite recursion when calling MaliciousNode.train
        Node.train(self)

    def train(self) -> None:
        self.y[self.source_indices] = self.source

        Node.train(self)
