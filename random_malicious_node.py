from node import Node
from utils import Weights
from attack import random_gaussian_weights


class RandomMaliciousNode(Node):
    mean: float
    sd: float

    def __init__(self,
                 mean: float = 0,
                 sd: float = 10,
                 attacking: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attacking = attacking
        self.mean = mean
        self.sd = sd

    def train(self) -> None:
        if not self.attacking:
            super().train()

    def get_weights(self) -> Weights:
        if self.attacking:
            return random_gaussian_weights(
                reference=self.model.get_weights(),
                mean=self.mean,
                sd=self.sd
            )
        else:
            return super().get_weights()
