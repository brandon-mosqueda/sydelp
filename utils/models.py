from keras import optimizers
from keras import layers, models

from utils.typing import KerasModel

def iris_model(learning_rate: float = 0.01) -> KerasModel:
    model: KerasModel = models.Sequential([
        layers.Input(shape=(4, )),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss='sparse_categorical_crossentropy'
    )

    return model


def mnist_model(learning_rate: float = 0.001,
                dense_units: int = 100,
                metrics: list = []) -> KerasModel:
    model: KerasModel = models.Sequential([
        layers.Input(shape=(784, )),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    return model


def spam_model(learning_rate: float = 0.001,
               vocabulary_size: int = 10000,
               sequence_length: int = 100,
               embedding_dim: int = 32,
               lstm_units: int = 32,
               metrics: list = []) -> KerasModel:
    model: KerasModel = models.Sequential([
        # The sequence length is the same as the number of columns in the input
        # matrix after tokenization
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(input_dim=vocabulary_size,
                         output_dim=embedding_dim),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss='binary_crossentropy',
        metrics=metrics
    )

    return model
