from learning.gossip import Gossip
from utils.utils import MyProgressBar, progress_bar
from nodes.sybilwall_node import SybilwallNode


class Sybilwall(Gossip[SybilwallNode]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialize neighbors
        for i, node in enumerate(self.nodes):
            node.id = i
            node.neighbors = {j: self.nodes[j] for j in self.graph.neighbors(i)}

    def gossip(self, iteration_num: int) -> None:
        print("\t+ Gossiping")
        bar: MyProgressBar = progress_bar(len(self.nodes) * 2)

        # First share the most recent version of own models with neighborhood
        for node in self.nodes:
            bar.next()

            for neighbor in node.neighbors.values():
                neighbor.replace_in_history(
                    node_id=node.id,
                    iteration_num=iteration_num,
                    distance=1,
                    model=node.own_history_weights,
                    sender_id=node.id,
                )

        for node in self.nodes:
            bar.next()
            node.gossip()

        bar.finish()

    def aggregation(self, iteration_num: int) -> None:
        # Update own local model histories
        for node in self.nodes:
            node.update_own_history()

        self.gossip(iteration_num)

        print("\t+ Aggregating")
        bar: MyProgressBar = progress_bar(len(self.nodes))

        # Aggregation
        for node in self.nodes:
            bar.next()
            node.aggregate()

        bar.finish()

        self.global_model_aggregation()
