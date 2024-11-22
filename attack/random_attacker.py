from attack.attacker import Attacker
from nodes.random_node import RandomNode
from utils.utils import NumArray, MyProgressBar
from typing import TypeVar


RandomNodeType = TypeVar('RandomNodeType', bound='RandomNode')


class RandomAttacker(Attacker[RandomNodeType]):
    def identical_attack(self,
                         attacking_nodes: list[RandomNodeType],
                         bar: MyProgressBar) -> None:
        attacking_nodes[0].attack()
        # The attack generates the random vector of parameters
        attack_model: list[NumArray] = attacking_nodes[0].get_model_params()

        for node in attacking_nodes:
            node.set_model_params(attack_model)
            bar.next()

        bar.finish()
