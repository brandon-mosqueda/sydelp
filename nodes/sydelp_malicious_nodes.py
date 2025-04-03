from nodes.sydelp_node import SydelpNode
from nodes.malicious_node import MaliciousNode
from nodes.random_node import RandomNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.untargeted_label_flipping_node import UntargetedLabelFlippingNode


class SydelpMaliciousNode(SydelpNode, MaliciousNode):
    pass


class SydelpRandomNode(SydelpNode, RandomNode):
    def attack(self) -> None:
        super().attack()
        self.momentum = self.flat_weights

class SydelpSignFlippingNode(SydelpNode, SignFlippingNode):
    def attack(self) -> None:
        super().attack()
        self.momentum = self.flat_weights


class SydelpTargetedLabelFlippingNode(SydelpNode, TargetedLabelFlippingNode):
    def attack(self) -> None:
        super().attack()
        self.momentum = self.flat_weights


class SydelpUntargetedLabelFlippingNode(SydelpNode, UntargetedLabelFlippingNode):
    def attack(self) -> None:
        super().attack()
        self.momentum = self.flat_weights
