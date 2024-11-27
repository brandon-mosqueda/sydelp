import numpy as np
import utils.utils as utils

from pandas import DataFrame, concat
from utils.typing import NumArray, IntArray, KerasModel, WeightsShapes, Int


class Node:
    x: NumArray
    y: IntArray
    model: KerasModel
    flat_weights: NumArray
    weights_shapes: WeightsShapes
    epochs: int
    batch_size: int
    rows_num: int
    is_malicious: bool = False

    def __init__(self,
                 x: NumArray,
                 y: IntArray,
                 model: KerasModel,
                 weights_shapes: WeightsShapes,
                 epochs: int,
                 batch_size: int) -> None:
        self.x = x
        self.y = y
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_shapes = weights_shapes

        self.rows_num = x.shape[0]

        total_size: Int = sum(np.prod(shape) for shape in weights_shapes)
        self.flat_weights = np.empty(total_size, dtype="float32")
        utils.set_weights_to_array(self.model.get_weights(),
                                   self.flat_weights)

    def set_model_weights(self, weights: list[NumArray]) -> None:
        self.model.set_weights(weights)
        utils.set_weights_to_array(weights, self.flat_weights)

    def set_flat_model_weights(self, flat_weights: NumArray) -> None:
        self.flat_weights[:] = flat_weights

        self.model.set_weights(utils.flat_weights_to_original(
            self.flat_weights,
            self.weights_shapes
        ))

    def get_model_weights(self) -> list[NumArray]:
        return self.model.get_weights()

    def get_flat_model_weights(self) -> NumArray:
        return self.flat_weights

    def train(self) -> None:
        self.model.fit(self.x,
                       self.y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)

        utils.set_weights_to_array(self.model.get_weights(),
                                   self.flat_weights)

    def predict(self, x: NumArray) -> DataFrame:
        preds: DataFrame = DataFrame(self.model.predict(x, verbose=0))

        # For binary classification
        if preds.shape[1] == 1:
            preds = concat([1 - preds[0], preds], axis=1)
            preds.columns = [0, 1]

        preds['predicted'] = np.argmax(preds, axis=1)

        return preds
