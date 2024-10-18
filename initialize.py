from split import class_non_iid_split, Split
from nodes.node import Node
from keras import optimizers
from keras import layers, models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import TypedDict, Callable, TypeVar, Generic
from numpy.typing import NDArray
from utils import NNumeric
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
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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


def mnist_model(learning_rate: float = 0.001) -> models.Model:
    model: models.Model = models.Sequential([
        layers.Input(shape=(784, )),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),  # type: ignore
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def init_mnist(nodes_num: int,
               learning_rate: float = 0.001,
               epochs: int = 5,
               batch_size: int = 10) -> Initializer:
    # Load and preprocess the data
    X_train: NDArray[NNumeric]
    X_test: NDArray[NNumeric]
    y_train: NDArray[NNumeric]
    y_test: NDArray[NNumeric]
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(60000, 784).astype("float32") / 255
    X_test = X_test.reshape(10000, 784).astype("float32") / 255

    global_model: models.Model = mnist_model(learning_rate)

    splits: list[Split] = class_non_iid_split(X_train, y_train, nodes_num)
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
