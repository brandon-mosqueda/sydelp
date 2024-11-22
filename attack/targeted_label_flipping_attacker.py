from numpy import concatenate
from attack.attacker import Attacker
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from utils.utils import NumArray, IntArray, MyProgressBar
from typing import TypeVar


TargetedLabelFlippingNodeType = TypeVar('TargetedLabelFlippingNodeType',
                                        bound='TargetedLabelFlippingNode')


# On this attack, we gather all the available data to train a model and then
# replicate it to all the sybil nodes
class TargetedLabelFlippingAttacker(Attacker[TargetedLabelFlippingNodeType]):
    x: NumArray
    y: IntArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.x = concatenate([node.x for node in self.nodes])
        self.y = concatenate([node.y for node in self.nodes])

    def identical_attack(self,
                         attacking_nodes: list[TargetedLabelFlippingNodeType],
                         bar: MyProgressBar) -> None:
        node: TargetedLabelFlippingNodeType = attacking_nodes[0]

        original_x: NumArray = node.x
        original_y: IntArray = node.y

        node.x = self.x
        node.y = self.y
        node.attack()

        attack_params: list[NumArray] = node.get_model_params()

        node.x = original_x
        node.y = original_y

        for node in attacking_nodes:
            node.set_model_params(attack_params)
            bar.next()

        bar.finish()
