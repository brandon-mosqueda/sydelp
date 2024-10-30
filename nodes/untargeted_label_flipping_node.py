from numpy import unique
from nodes.malicious_node import MaliciousNode
from utils.attack import flip_labels


# When attacking the model is normally trained but using the flipped labels
class UntargetedLabelFlippingNode(MaliciousNode):
    has_real_labels: bool = True
    classes_num: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.classes_num = len(unique(self.y))

    def train(self) -> None:
        # Attacking | Has_real | Change labels
        #   False   | False    |    True        to original
        #   False   | True     |    False
        #   True    | False    |    False
        #   True    | True     |    True        to flipped
        if ((not self.attacking and not self.has_real_labels) or
            (self.attacking and self.has_real_labels)):
            # When applied to the flipped labels it will return to the original
            # label
            self.y = self.classes_num - self.y - 1

        self.has_real_labels = not self.attacking

        super().train()
