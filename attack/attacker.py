from abc import ABC, abstractmethod
from nodes.malicious_node import MaliciousNode
from typing import Generic, TypeVar
from utils.utils import MyProgressBar


MalNodeType = TypeVar('MalNodeType', bound='MaliciousNode')

# This class orchestrates the attacker. In case of an identical attack, it will
# create the attack model and replicate it to all sybils.
class Attacker(ABC, Generic[MalNodeType]):
    is_identical_attack: bool
    nodes: list[MalNodeType]

    def __init__(self,
                 nodes: list[MalNodeType],
                 is_identical_attack: bool) -> None:
        self.nodes = nodes
        self.is_identical_attack = is_identical_attack

    @abstractmethod
    def identical_attack(self,
                         attacking_nodes: list[MalNodeType],
                         bar: MyProgressBar) -> None:
        pass

    def attack(self) -> None:
        if not self.nodes:
            return

        bar: MyProgressBar = MyProgressBar(len(self.nodes))
        attacking_nodes: list[MalNodeType] = [node
                                              for node in self.nodes
                                              if node.attacking]
        not_attacking_nodes: list[MalNodeType] = [node
                                                  for node in self.nodes
                                                  if not node.attacking]

        for node in not_attacking_nodes:
            node.train()
            bar.next()

        if not attacking_nodes:
            return

        if self.is_identical_attack:
            self.identical_attack(attacking_nodes, bar)
        else:
            for node in attacking_nodes:
                node.attack()
                bar.next()

            bar.finish()
