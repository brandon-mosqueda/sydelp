import numpy as np

from attack.attacker import Attacker
from nodes.sydelp_malicious_nodes import SydelpMaliciousNode
from utils.typing import Float, FloatArray


class SydelpAttacker(Attacker[SydelpMaliciousNode]):
    computing_power: int # P
    is_worst_case: bool

    def __init__(self,
                 computing_power: int,
                 is_worst_case: bool,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            # to keep them all. We will remove as many nodes as necessary starting
            # with those with highest difficulty (which are usually the newer
            # ones)
            order_diff_idx: list[int] = np.flip(
                np.argsort(difficulties)
            ).tolist()
            remove_idx: list[int] = []

            while total_difficulties > self.computing_power:
                idx = order_diff_idx.pop()
                total_difficulties -= difficulties[idx]
                remove_idx.append(idx)

            # We have to remove them like this to ensure the reference in
            # Learning.mal_nodes is also updated
            for idx in remove_idx:
                del self.nodes[idx]
        else:
            new_nodes_num: int = int(self.computing_power - total_difficulties)

            for i in range(new_nodes_num):
                self.nodes.append(self.nodes[0].clone())
