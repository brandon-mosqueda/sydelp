import numpy as np

from nodes.malicious_node import MaliciousNode
from utils.utils import MyProgressBar, progress_bar
from utils.typing import *
from typing import TypeVar, Generic

MaliciousNodeType = TypeVar('MaliciousNodeType', bound='MaliciousNode')

# This class orchestrates all the malicious nodes
class Attacker(Generic[MaliciousNodeType]):
    is_identical_attack: bool
    nodes: list[MaliciousNodeType]
    x: NumArray
    y: IntArray

    def __init__(self,
                 nodes: list[MaliciousNodeType],
                 is_identical_attack: bool) -> None:
        self.nodes = nodes
        self.is_identical_attack = is_identical_attack

        self.x = np.concatenate([node.x for node in self.nodes])
        self.y = np.concatenate([node.y for node in self.nodes])

    def identical_attack(self) -> None:
        bar: MyProgressBar = progress_bar(len(self.nodes))

        node: MaliciousNodeType = self.nodes[0]

        original_x: NumArray = node.x
        original_y: IntArray = node.y

        # For those attacks that require a training dataset (e.g. Label
        # flipping), we use the whole dataset to fit the model and then we
        # replicate it
        node.x = self.x
        node.y = self.y
        node.attack()

        attack_params: list[FloatArray] = node.get_model_weights()

        # We gave him back his dataset to not interfer with other parts
        node.x = original_x
        node.y = original_y

        for node in self.nodes:
            node.set_model_weights(attack_params)
            bar.next()

        bar.finish()

    def attack(self) -> None:
        if self.is_identical_attack:
            self.identical_attack()
            return

        bar: MyProgressBar = progress_bar(len(self.nodes))

        for node in self.nodes:
            node.attack()
            bar.next()

        bar.finish()
