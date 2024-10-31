import pandas as pd
import requests as rq

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from os.path import isfile
from zipfile import ZipFile
from io import BytesIO
from utils.split import class_non_iid_split, Split, dirichlet_split
from nodes.node import Node
from keras import optimizers
from keras import layers, models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import TypedDict, Callable, TypeVar, Union
from numpy.typing import NDArray
from utils.utils import NNumeric
from keras.datasets.mnist import load_data as load_mnist  # type: ignore


class Initializer(TypedDict):
    nodes: list[Node]
    global_model: models.Model
    X_test: NDArray[NNumeric]
    y_test: NDArray[NNumeric]


AnyNode = TypeVar('AnyNode', bound='Node')

def init_nodes(splits: list[Split],
               model_fn: Callable,
               learning_rate: float = 0.01,
               epochs: int = 10,
               batch_size: int = 32,
               node_class: Callable[..., AnyNode] = Node,
               **kwargs) -> list[AnyNode]:
    nodes: list[AnyNode] = []

    for split in splits:
        model: models.Model = model_fn(learning_rate)
        node: AnyNode = node_class(x=split['X'],
                                   y=split['y'],
                                   model=model,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   **kwargs)
        nodes.append(node)

    return nodes


def iris_model(learning_rate: float = 0.01) -> models.Model:
    model: models.Model = models.Sequential([
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


def iris_data(testing_proportion: float = 0.2) -> list[NDArray[NNumeric]]:
    # Load and preprocess the data
    X: NDArray[NNumeric]; y: NDArray[NNumeric]
    X, y = load_iris(return_X_y=True)  # type: ignore

    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]

    # Split the data into balanced training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testing_proportion, stratify=y
    )

    # Standardize the features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, y_train, y_test]


def init_iris(nodes_num: int,
              testing_proportion: float = 0.2,
              learning_rate: float = 0.01,
              epochs: int = 10,
              batch_size: int = 32) -> Initializer:
    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]

    # Split the data into balanced training and testing datasets
    X_train, X_test, y_train, y_test = iris_data(testing_proportion)

    global_model: models.Model = iris_model(learning_rate)

    splits: list[Split] = class_non_iid_split(X_train, y_train, nodes_num)
    nodes: list[Node] = init_nodes(splits=splits,
                                   model_fn=iris_model,
                                   learning_rate=learning_rate,
                                   epochs=epochs,
                                   batch_size=batch_size)

    return {
        'nodes': nodes,
        'global_model': global_model,
        'X_test': X_test,
        'y_test': y_test
    }


def mnist_model(learning_rate: float = 0.001,
                dense_units: int = 100,
                metrics: list = []) -> models.Model:
    model: models.Model = models.Sequential([
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


def mnist_data() -> list[NDArray[NNumeric]]:
    # Load and preprocess the data
    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(60000, 784).astype("float32") / 255
    X_test = X_test.reshape(10000, 784).astype("float32") / 255

    return [X_train, X_test, y_train, y_test]


def init_mnist(nodes_num: int,
               learning_rate: float = 0.001,
               dense_units: int = 100,
               epochs: int = 5,
               batch_size: int = 16,
               alpha: float = 0.1,
               split_min_size: Union[int, None] = 16) -> Initializer:
    # Load and preprocess the data
    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]
    X_train, X_test, y_train, y_test = mnist_data()

    global_model: models.Model = mnist_model(learning_rate, dense_units)

    splits: list[Split] = dirichlet_split(
        X_train,
        y_train,
        n_splits=nodes_num,
        alpha=alpha,
        split_min_size=split_min_size
    )
    nodes: list[Node] = init_nodes(splits=splits,
                                   model_fn=mnist_model,
                                   learning_rate=learning_rate,
                                   epochs=epochs,
                                   batch_size=batch_size)

    return {
        'nodes': nodes,
        'global_model': global_model,
        'X_test': X_test,
        'y_test': y_test
    }


def spam_model(learning_rate: float = 0.01,
               vocabulary_size: int = 1000,
               embedding_dim: int = 64,
               lstm_units: int = 32) -> models.Model:
    model = Sequential([
        layers.Embedding(input_dim=vocabulary_size,
                         output_dim=embedding_dim),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss='binary_crossentropy'
    )

    return model


def spam_data(testing_proportion: float = 0.2,
              vocabulary_size: int = 1000,
              max_sequence_length: int = 50) -> list[NDArray[NNumeric]]:
    file_path: str = "data/sms_spam_collection/SMSSpamCollection"

    if not isfile(file_path):
        url: str = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00228/smsspamcollection.zip"
        )
        response = rq.get(url)
        if response.status_code == 200:
            with ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall("data/sms_spam_collection")
        else:
            raise ConnectionError("Failed to download file. Status code: %s" %
                                  response.status_code)

    Data: pd.DataFrame = pd.read_csv(file_path,
                                     sep='\t',
                                     header=None,
                                     names=['label', 'text'],
                                     encoding='latin-1')
    Data['label'] = LabelEncoder().fit_transform(Data['label'])

    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]

    X_train, X_test, y_train, y_test = train_test_split(
        Data['text'],
        Data['label'].to_numpy(),
        stratify=Data['label'],
        test_size=testing_proportion
    )

    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_padded: NDArray[NNumeric] = pad_sequences(
        X_train_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )
    X_test_padded: NDArray[NNumeric] = pad_sequences(
        X_test_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    return [X_train_padded, X_test_padded, y_train, y_test]
