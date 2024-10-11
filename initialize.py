from split import class_non_iid_split, Split
from node import Node
from keras import optimizers
from keras import layers, models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils._bunch import Bunch
from typing import TypedDict, Callable
from numpy.typing import NDArray
from utils import NNumeric

class Initializer(TypedDict):
    nodes: list[Node]
    global_model: models.Model
    X_test: NDArray[NNumeric]
    y_test: NDArray[NNumeric]

def init_nodes(splits: list[Split],
               model_fn: Callable,
               learning_rate: float = 0.01,
               epochs: int = 10,
               batch_size: int = 32) -> list[Node]:
    nodes: list[Node] = []

    for split in splits:
        model: models.Model = model_fn(learning_rate)
        node: Node = Node(x=split['X'],
                          y=split['y'],
                          model=model,
                          epochs=epochs,
                          batch_size=batch_size)
        nodes.append(node)

    return nodes

# Define a simple neural network model
def iris_model(learning_rate: float = 0.01) -> models.Model:
    model: models.Model = models.Sequential([
        layers.Input(shape=(4, )),
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate), # type: ignore
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

def init_iris(nodes_num: int,
              testing_proportion: float = 0.2,
              learning_rate: float = 0.01,
              epochs: int = 10,
              batch_size: int = 32) -> Initializer:
    # Load and preprocess the data
    X: NDArray[NNumeric]; y: NDArray[NNumeric]
    X, y = load_iris(return_X_y=True) # type: ignore

    X_train: NDArray[NNumeric]; X_test: NDArray[NNumeric];
    y_train: NDArray[NNumeric]; y_test: NDArray[NNumeric]

    # Split the data into balanced training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testing_proportion, stratify=y
    )

    # Standardize the features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
