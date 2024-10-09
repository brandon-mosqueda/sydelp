import numpy as np

from utils import NNumeric
from numpy.typing import NDArray
from pandas import DataFrame
from keras.src.models import Model as KerasModel

class Node:
    x: NDArray[NNumeric]
    y: NDArray[NNumeric]
    model: KerasModel
    epochs: int
    batch_size: int
    rows_num: int

    def __init__(self,
                 x: NDArray[NNumeric],
                 y: NDArray[NNumeric],
                 model: KerasModel,
                 epochs: int,
                 batch_size: int) -> None:
        self.x = x
        self.y = y
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        self.rows_num = x.shape[0]

    def set_weights(self, weights: list[NDArray[NNumeric]]) -> None:
        self.model.set_weights(weights)

    def get_weights(self) -> list[NDArray[NNumeric]]:
        return self.model.get_weights()

    def train(self) -> None:
        self.model.fit(self.x,
                       self.y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)

    def predict(self, x) -> DataFrame:
        predictions: DataFrame = DataFrame(self.model.predict(x, verbose=0))
        predictions["predicted"] = np.argmax(predictions, axis=1)

        return predictions
