from numpy import concatenate
from attack.attacker import Attacker
from nodes.sign_flipping_node import SignFlippingNode
from utils.utils import NumArray, IntArray, MyProgressBar
from typing import TypeVar


SignFlippingNodeType = TypeVar('SignFlippingNodeType', bound='SignFlippingNode')


# On this attack, we gather all the available data to train a model and then
# replicate it to all the sybil nodes
class SignFlippingAttacker(Attacker[SignFlippingNodeType]):
    x: NumArray
    y: IntArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.x = concatenate([node.x for node in self.nodes])
        self.y = concatenate([node.y for node in self.nodes])

    def identical_attack(self,
                         attacking_nodes: list[SignFlippingNodeType],
                         bar: MyProgressBar) -> None:
        node: SignFlippingNodeType = attacking_nodes[0]

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
