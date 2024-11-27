from numpy.random import normal as random_gauss
from nodes.malicious_node import MaliciousNode


# This attacks consist on skipping the train phase and send random gaussian
# vectors for aggregation. We need the mean and the standard deviation.
class RandomNode(MaliciousNode):
    mean: float
    sd: float

    def __init__(self,
                 mean: float = 0,
                 sd: float = 10,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.sd = sd

    def attack(self) -> None:
        self.set_flat_model_weights(random_gauss(
            loc=self.mean,
            scale=self.sd,
            size=self.flat_weights.size
        ))
