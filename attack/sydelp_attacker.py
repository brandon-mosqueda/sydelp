import numpy as np

from attack.attacker import Attacker
from nodes.sydelp_malicious_nodes import SydelpMaliciousNode
from utils.utils import MyProgressBar, progress_bar
from utils.typing import Float, FloatArray
from utils.split import Split, balanced_split


class SydelpAttacker(Attacker[SydelpMaliciousNode]):
    computing_power: int # P
    is_worst_case: bool
    # This is the number of desired malicious nodes (necessary to break the
    # security guarantees on aggregation).
    objective_malicious_num: int

    def __init__(self,
                 computing_power: int,
                 is_worst_case: bool,
                 objective_malicious_num: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.objective_malicious_num = objective_malicious_num
        self.computing_power = computing_power
        self.is_worst_case = is_worst_case

    # This method is always called after aggregation, so the contribution scores
    # are updated for the current iteration.
    def update_nodes(self) -> None:
        # Only the worst case attack is dynamic
        if not self.is_worst_case:
            return

        difficulties: FloatArray = np.array([node.compute_difficulty()
                                             for node in self.nodes],
                                             dtype="float")
        total_difficulties: Float = difficulties.sum()

        if total_difficulties > self.computing_power:
            # Some nodes where penalized and there is no enough computing power
            # to keep them all. We will remove as many nodes as necessary
            # starting with those with highest difficulty (which are usually the
            # newer ones)
            order_diff_idx: list[int] = np.flip(
                np.argsort(difficulties)
            ).tolist()
            remove_nodes: list[SydelpMaliciousNode] = []

            while total_difficulties > self.computing_power:
                idx = order_diff_idx.pop()
                total_difficulties -= difficulties[idx]
                remove_nodes.append(self.nodes[idx])

            # We have to remove them like this (by reference) to ensure the
            # reference in Learning.mal_nodes is also updated
            for node in remove_nodes:
                self.nodes.remove(node)
        else:
            new_nodes_num: int = int(self.computing_power - total_difficulties)

            for _ in range(new_nodes_num):
                self.nodes.append(self.nodes[0].clone())

            # If there are new nodes, the malicious dataset is reassigned
            # equally to all of them.
            if new_nodes_num > 0:
                splits: list[Split] = balanced_split(self.x,
                                                     self.y,
                                                     n_splits=len(self.nodes))

                for split, node in zip(splits, self.nodes):
                    node.set_new_dataset(split["X"], split["y"])

    def attack(self) -> None:
        if (self.is_worst_case
            and len(self.nodes) < self.objective_malicious_num):

            bar: MyProgressBar = progress_bar(len(self.nodes))

            for node in self.nodes:
                bar.next()
                node.train()

            bar.finish()
        else:
            super().attack()

            # In the identical attack, the malicious weights are not assigned
            # to the momentum vector because the identical attack method is
            # called in the attacker class where normal malicious nodes
            # (without) momentum are used. In non-identical attack, this does
            # not occur as each node attacks separately.
            if self.is_identical_attack:
                for node in self.nodes:
                    node.momentum[:] = self.nodes[0].flat_weights
