from nodes.malicious_node import MaliciousNode
from attack_utils import flip_labels


# When attacking the model is normally trained but using the flipped labels
class TargetedLabelFlippingNode(MaliciousNode):
    has_real_labels: bool
    target_label_1: int
    target_label_2: int

    def __init__(self,
                 target_label_1: int,
                 target_label_2: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_label_1 = target_label_1
        self.target_label_2 = target_label_2
        self.has_real_labels = True

    def train(self) -> None:
        # Attacking | Has_real | Change labels
        #   False   | False    |    True        to original
        #   False   | True     |    False
        #   True    | False    |    False
        #   True    | True     |    True        to flipped
        if ((not self.attacking and not self.has_real_labels) or
            (self.attacking and self.has_real_labels)):
            flip_labels(self.y, self.target_label_1, self.target_label_2)

        self.has_real_labels = not self.attacking

        super().train()
