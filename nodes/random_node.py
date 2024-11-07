from nodes.malicious_node import MaliciousNode
from utils.utils import ModelParams
from utils.attack import random_gaussian_model


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

    def train(self) -> None:
        if self.attacking:
            self.set_model_params(random_gaussian_model(
                reference=self.model.get_weights(),
                mean=self.mean,
                sd=self.sd
            ))
        else:
            super().train()
