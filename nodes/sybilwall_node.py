import numpy as np
from typing import TypedDict, Union
from utils.typing import NumArray, IntArray
from nodes.node import Node


class HistoricModel(TypedDict):
    node_id: int       # p
    model: NumArray    # h
    iteration_num: int # r
    distance: int      # d
    sender_id: int     # f


class SybilwallNode(Node):
    history: dict[int, HistoricModel]
    predict_batch_size: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history = {}
        self.predict_batch_size = 252

    # model should be a copy, not a reference
    def add_in_history(self,
                       node_id: int,
                       model: NumArray,
                       iteration_num: int,
                       distance: int,
                       sender_id: int) -> None:
        hist: Union[None, HistoricModel] = self.history.get(node_id, None)

        if hist is None:
            self.history[node_id] = {
                'node_id': node_id,
                'model': model.copy(),
                'iteration_num': iteration_num,
                'distance': distance,
                'sender_id': sender_id,
            }
        elif iteration_num > hist['iteration_num']:
            hist['model'] += model
            hist['iteration_num'] = iteration_num
            hist['distance'] = distance
            hist['sender_id'] = sender_id

    # model should be a copy, not a reference
    def replace_in_history(self,
                           node_id: int,
                           model: NumArray,
                           iteration_num: int,
                           distance: int,
                           sender_id: int):
        hist: Union[None, HistoricModel] = self.history.get(node_id, None)

        if hist is None:
            self.history[node_id] = {
                'node_id': node_id,
                'model': model.copy(),
                'iteration_num': iteration_num,
                'distance': distance,
                'sender_id': sender_id,
            }
        elif iteration_num > hist['iteration_num']:
            hist['model'][:] = model
            hist['iteration_num'] = iteration_num
            hist['distance'] = distance
            hist['sender_id'] = sender_id

    def predict(self, x: NumArray) -> IntArray:
        predicted: NumArray = self.model.predict(
            x,
            verbose=0,
            batch_size=self.predict_batch_size
        )

        # For binary classification
        if predicted.shape[1] == 1:
            return (predicted > 0.5).flatten().astype(int)

        return np.argmax(predicted, axis=1)
