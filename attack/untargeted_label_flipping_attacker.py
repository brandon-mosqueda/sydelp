from numpy import concatenate
from attack.attacker import Attacker
from nodes.untargeted_label_flipping_node import UntargetedLabelFlippingNode
from utils.typing import NumArray, IntArray
from typing import TypeVar


UntargetedLabelFlippingNodeType = TypeVar('UntargetedLabelFlippingNodeType',
                                          bound='UntargetedLabelFlippingNode')


# On this attack, we gather all the available data to train a model and then
# replicate it to all the sybil nodes
class UntargetedLabelFlippingAttacker(Attacker[UntargetedLabelFlippingNodeType]):
    x: NumArray
    y: IntArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.x = concatenate([node.x for node in self.nodes])
        self.y = concatenate([node.y for node in self.nodes])

    def get_attack_params(self,
                          attacking_nodes: list[UntargetedLabelFlippingNodeType]) -> list[NumArray]:
        node: UntargetedLabelFlippingNodeType = attacking_nodes[0]

        original_x: NumArray = node.x
        original_y: IntArray = node.y

        node.x = self.x
        node.y = self.y
        node.attack()

        node.x = original_x
        node.y = original_y

        return node.get_model_weights()
