from numpy.random import normal as random_gauss
from nodes.malicious_node import MaliciousNode
from utils.utils import NumArray, flatten_to_original


# This attacks consist on skipping the train phase and send random gaussian
# vectors for aggregation. We need the mean and the standard deviation.
class RandomNode(MaliciousNode):
    model_size: int
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
        self.model_size = sum(w.size for w in self.get_model_params())

    def attack(self) -> None:
        random_model: list[NumArray] = flatten_to_original(
            random_gauss(loc=self.mean, scale=self.sd, size=self.model_size),
            self.model
        )

        self.set_model_params(random_model)
