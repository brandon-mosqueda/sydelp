from utils import Weights
from nodes.malicious_node import MaliciousNode


# When attacking the model is normally trained but using the flipped labels
class SignFlippingNode(MaliciousNode):
    scale_factor: float

    def __init__(self,
                 scale_factor: float = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor

    def train(self) -> None:
        super().train()

        if self.attacking:
            # Scale and flip sign
            self.set_weights(
                [-self.scale_factor * weigth
                 for weigth in self.get_weights()]
            )
