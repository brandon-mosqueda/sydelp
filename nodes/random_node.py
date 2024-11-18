from nodes.malicious_node import MaliciousNode
from utils.attack import random_gaussian_model
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
        self.model_size = sum(w.size for w in self.model.get_weights())

    def attack(self) -> None:
        random_model: list[NumArray] = flatten_to_original(
            random_gaussian_model(
                size=self.model_size,
                mean=self.mean,
                sd=self.sd
            ),
            self.model
        )

        self.set_model_params(random_model)
