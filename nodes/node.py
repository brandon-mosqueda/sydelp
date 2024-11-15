from utils.utils import NumArray, IntArray, get_flatten_weights, set_flatten_weights
from keras.src.models import Model as KerasModel


class Node:
    x: NumArray
    y: IntArray
    model: KerasModel
    epochs: int
    batch_size: int
    rows_num: int
    is_malicious: bool = False

    def __init__(self,
                 x: NumArray,
                 y: IntArray,
                 model: KerasModel,
                 epochs: int,
                 batch_size: int) -> None:
        self.x = x
        self.y = y
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        self.rows_num = x.shape[0]

    def set_model_params(self, model_params: NumArray) -> None:
        set_flatten_weights(self.model, model_params)

    def get_model_params(self) -> NumArray:
        return get_flatten_weights(self.model)

    def train(self) -> None:
        self.model.fit(self.x,
                       self.y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)
