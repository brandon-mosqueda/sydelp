from nodes.sybilwall_node import SybilwallNode
from nodes.malicious_node import MaliciousNode
from nodes.random_node import RandomNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.untargeted_label_flipping_node import UntargetedLabelFlippingNode


class SybilwallMaliciousNode(SybilwallNode, MaliciousNode):
    pass


class SybilwallRandomNode(SybilwallNode, RandomNode):
    pass


class SybilwallSignFlippingNode(SybilwallNode, SignFlippingNode):
    pass


class SybilwallTargetedLabelFlippingNode(SybilwallNode,
                                         TargetedLabelFlippingNode):
    pass


class SybilwallUntargetedLabelFlippingNode(SybilwallNode,
                                           UntargetedLabelFlippingNode):
    pass
