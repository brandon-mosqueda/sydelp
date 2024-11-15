from abc import ABC, abstractmethod
from nodes.node import Node


class MaliciousNode(Node, ABC):
    attacking: bool
    is_malicious: bool = True

    def __init__(self, attacking: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attacking = attacking

    @abstractmethod
    def attack(self) -> None:
        pass

    def train(self) -> None:
        if self.attacking:
            self.attack()
        else:
            super().train()
