from abc import ABC, abstractmethod
from nodes.node import Node
from utils.typing import NumArray, IntArray


class MaliciousNode(Node, ABC):
    is_malicious: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def attack(self) -> None:
        pass

    def set_new_dataset(self, x: NumArray, y: IntArray) -> None:
        self.x = x
        self.y = y
