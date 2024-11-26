from typing import TypedDict, Union
from utils.utils import NumArray
from nodes.node import Node


class HistoricModel(TypedDict):
    node_id: int       # p
    model: NumArray    # h
    iteration_num: int # r
    distance: int      # d
    sender_id: int     # f


class SybilwallNode(Node):
    history: dict[int, HistoricModel]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history = {}

    def add_in_history(self, model: HistoricModel) -> None:
        hist: Union[None, HistoricModel] = self.history.get(model['node_id'],
                                                            None)
        if hist is None:
            self.history[model['node_id']] = model
        elif model['iteration_num'] > hist['iteration_num']:
            model['model'] += hist['model']
            self.history[model['node_id']] = model

    def replace_in_history(self, model: HistoricModel):
        hist: Union[None, HistoricModel] = self.history.get(model['node_id'],
                                                            None)

        # If it has no previous information of that node or it is a more updated
        # version
        if hist is None or model['iteration_num'] > hist['iteration_num']:
            self.history[model['node_id']] = model
