from node import Node
from utils import Weights
from attack_utils import random_gaussian_weights


# This attacks consist on skipping the train phase and send random gaussian
# vectors for aggregation. We need the mean and the standard deviation.
class RandomMaliciousNode(Node):
    attacking: bool
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
