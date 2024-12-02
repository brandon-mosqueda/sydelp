from numpy import array
from utils.aggregation import fed_avg
from utils.typing import FloatArray

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
    # This method is overwritten here because otherwise, the weights of
    # malicious nodes become huge and end up overflowing
    def aggregate(self):
        models: FloatArray = array([
            neighbor.current_weights
            for neighbor in self.neighbors.values()
            if not neighbor.is_malicious
        ])

        if len(models):
            self.set_flat_model_weights(fed_avg(models))


class SybilwallTargetedLabelFlippingNode(SybilwallNode,
                                         TargetedLabelFlippingNode):
    pass


class SybilwallUntargetedLabelFlippingNode(SybilwallNode,
                                           UntargetedLabelFlippingNode):
    pass
