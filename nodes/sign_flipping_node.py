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

    def attack(self) -> None:
        super().train()

        # Scale and flip sign
        self.set_model_params(
            [-self.scale_factor * param for param in self.get_model_params()])
