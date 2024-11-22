from nodes.mab_node import MabNode
from nodes.malicious_node import MaliciousNode
from nodes.random_node import RandomNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.untargeted_label_flipping_node import UntargetedLabelFlippingNode


class MabMaliciousNode(MabNode, MaliciousNode):
    pass


class MabRandomNode(MabNode, RandomNode):
    pass


class MabSignFlippingNode(MabNode, SignFlippingNode):
    pass


class MabTargetedLabelFlippingNode(MabNode, TargetedLabelFlippingNode):
    pass


class MabUntargetedLabelFlippingNode(MabNode, UntargetedLabelFlippingNode):
    pass
