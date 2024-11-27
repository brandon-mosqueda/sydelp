import utils.utils as utils

from pandas import DataFrame, concat
from numpy import argmax
from utils.typing import NumArray, IntArray, KerasModel, ModelSizes


class Node:
    x: NumArray
    y: IntArray
    model: KerasModel
    flatten_weights: NumArray
    model_sizes: ModelSizes
    epochs: int
    batch_size: int
    rows_num: int
    is_malicious: bool = False

    def __init__(self,
                 x: NumArray,
                 y: IntArray,
                 model: KerasModel,
                 epochs: int,
                 batch_size: int,
                 model_sizes: ModelSizes) -> None:
        self.x = x
        self.y = y
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_sizes = model_sizes

        self.rows_num = x.shape[0]
        self.flatten_weights = utils.get_flatten_weights(self.model)

    def set_model_params(self, model_params: list[NumArray]) -> None:
        self.model.set_weights(model_params)
        utils.set_list_weights_in_array(model_params, self.flatten_weights)

    def set_flatten_model_params(self, model_params: NumArray) -> None:
        self.flatten_weights[:] = model_params
        self.model.set_weights(utils.flatten_to_original(model_params, self.model))

    def get_model_params(self) -> list[NumArray]:
        return self.model.get_weights()

    def get_flatten_model_params(self) -> NumArray:
        return utils.get_flatten_weights(self.model)

    def train(self) -> None:
        self.model.fit(self.x,
                       self.y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)

    def predict(self, x: NumArray) -> DataFrame:
        preds: DataFrame = DataFrame(self.model.predict(x, verbose=0))

        # For binary classification
        if preds.shape[1] == 1:
            preds = concat([1 - preds[0], preds], axis=1)
            preds.columns = [0, 1]

        preds['predicted'] = argmax(preds, axis=1)

        return preds
