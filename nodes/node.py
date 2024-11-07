from utils.utils import NNumeric, ModelParams
from numpy.typing import NDArray
from keras.src.models import Model as KerasModel


class Node:
    x: NDArray[NNumeric]
    y: NDArray[NNumeric]
    model: KerasModel
    epochs: int
    batch_size: int
    rows_num: int
    is_malicious: bool = False

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

    def set_model_params(self, model_params: ModelParams) -> None:
        self.model.set_weights(model_params)

    def get_model_params(self) -> ModelParams:
        return self.model.get_weights()

    def train(self) -> None:
        self.model.fit(self.x,
                       self.y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)
