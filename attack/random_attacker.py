from attack.attacker import Attacker
from nodes.random_node import RandomNode
from utils.typing import FloatArray
from typing import TypeVar


RandomNodeType = TypeVar('RandomNodeType', bound='RandomNode')


class RandomAttacker(Attacker[RandomNodeType]):
    def get_attack_params(self,
                          attacking_nodes: list[RandomNodeType]) -> list[FloatArray]:
        # The attack generates the random vector of parameters
        attacking_nodes[0].attack()
        return attacking_nodes[0].get_model_weights()
