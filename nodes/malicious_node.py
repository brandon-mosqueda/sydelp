from abc import ABC, abstractmethod
from nodes.node import Node


class MaliciousNode(Node, ABC):
    is_malicious: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def attack(self) -> None:
        pass
