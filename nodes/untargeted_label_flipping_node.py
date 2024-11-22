from numpy import unique
from nodes.malicious_node import MaliciousNode


# When attacking the model is normally trained but using the flipped labels
class UntargetedLabelFlippingNode(MaliciousNode):
    has_real_labels: bool = True
    classes_num: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.classes_num = len(unique(self.y))

    def flip_labels(self) -> None:
        self.y = self.classes_num - self.y - 1

    def attack(self) -> None:
        if self.has_real_labels:
            self.flip_labels()
            self.has_real_labels = False

        super().train()

    def train(self) -> None:
        if not self.has_real_labels:
            self.flip_labels()
            self.has_real_labels = True

        super().train()
