from __future__ import annotations

from abc import ABC, abstractmethod

from utils.utils import replicate_model

from nodes.sydelp_node import SydelpNode
from nodes.malicious_node import MaliciousNode
from nodes.random_node import RandomNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.untargeted_label_flipping_node import UntargetedLabelFlippingNode


class SydelpMaliciousNode(SydelpNode, MaliciousNode, ABC):
    def attack(self) -> None:
        super().attack()
        self.momentum[:] = self.flat_weights

    @abstractmethod
    def add_specific_params(self, base_params: dict) -> dict:
        pass

    def clone_parameters(self) -> dict:
        params: dict = {
            "x": self.x,
            "y": self.y,
            "weights_shapes": self.weights_shapes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "momentum_coeff": self.momentum_coeff,
            "difficulty_alpha": self.difficulty_alpha,
            "iterations_num": self.iterations_num,
            "model": replicate_model(self.model, 1)[0],
        }

        params = self.add_specific_params(params)

        return params

    @abstractmethod
    def clone(self) -> SydelpMaliciousNode:
        pass

class SydelpRandomNode(SydelpMaliciousNode, RandomNode):
    def add_specific_params(self, base_params: dict) -> dict:
        base_params['mean'] = self.mean
        base_params['sd'] = self.sd

        return base_params

    def clone(self) -> SydelpRandomNode:
        params: dict = self.clone_parameters()

        return SydelpRandomNode(**params)

class SydelpSignFlippingNode(SydelpMaliciousNode, SignFlippingNode):
    def add_specific_params(self, base_params: dict) -> dict:
        base_params['scale_factor'] = self.scale_factor

        return base_params

    def clone(self) -> SydelpSignFlippingNode:
        params: dict = self.clone_parameters()

        return SydelpSignFlippingNode(**params)


class SydelpTargetedLabelFlippingNode(SydelpMaliciousNode, TargetedLabelFlippingNode):
    def add_specific_params(self, base_params: dict) -> dict:
        base_params['source'] = self.source
        base_params['target'] = self.target

        return base_params

    def clone(self) -> SydelpTargetedLabelFlippingNode:
        params: dict = self.clone_parameters()

        return SydelpTargetedLabelFlippingNode(**params)


class SydelpUntargetedLabelFlippingNode(SydelpMaliciousNode, UntargetedLabelFlippingNode):
    def add_specific_params(self, base_params: dict) -> dict:
        return base_params

    def clone(self) -> SydelpTargetedLabelFlippingNode:
        params: dict = self.clone_parameters()

        return SydelpTargetedLabelFlippingNode(**params)
