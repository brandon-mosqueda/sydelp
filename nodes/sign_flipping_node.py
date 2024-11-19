from nodes.node import Node
from nodes.malicious_node import MaliciousNode
from utils.utils import NumArray


# When attacking the model is normally trained but using the flipped labels
class SignFlippingNode(MaliciousNode):
    scale_factor: float

    def __init__(self,
                 scale_factor: float = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor

    def attack(self) -> None:
        # This prevents infinite recursion when calling MaliciousNode.train
        Node.train(self)

        # Scale and flip sign
        self.set_model_params([
            -self.scale_factor * layer_w
            for layer_w in self.model.get_weights()
        ])
