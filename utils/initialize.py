import keras
import pandas as pd
import requests as rq
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
from os.path import isfile
from zipfile import ZipFile
from io import BytesIO
from nodes.node import Node
from keras import optimizers
from keras import layers, models
from keras.src.models import Model as KerasModel
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import TypedDict, Union
from utils.utils import NumArray, IntArray, as_name
from utils.split import Split, balanced_split, dirichlet_split
from keras.src.models import Model as KerasModel
from learning.learning import MetricParams, Learning
from learning.federated import FederatedLearning
from learning.sydelp import Sydelp
from learning.mab import Mab
from utils.metrics import f1_score, label_flipping_success_rate, label_recall
from sklearn.metrics import accuracy_score

from attack.attacker import Attacker
from attack.random_attacker import RandomAttacker
from attack.sign_flipping_attacker import SignFlippingAttacker
from attack.targeted_label_flipping_attacker import TargetedLabelFlippingAttacker
from attack.untargeted_label_flipping_attacker import UntargetedLabelFlippingAttacker

from nodes.node import Node
from nodes.random_node import RandomNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.mab_malicious_nodes import *
from nodes.mab_node import MabNode

load_mnist = keras.datasets.mnist.load_data

class Initializer(TypedDict):
    nodes: list[Node]
    global_model: KerasModel
    X_test: NumArray
    y_test: IntArray


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


def iris_data(testing_proportion: float = 0.2) -> tuple[NumArray, NumArray,
                                                        IntArray, IntArray]:
    # Load and preprocess the data
    X: NumArray; y: IntArray
    X, y = load_iris(return_X_y=True)  # type: ignore

    X_train: NumArray; X_test: NumArray
    y_train: IntArray; y_test: IntArray

    # Split the data into balanced training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testing_proportion, stratify=y
    )

    # Standardize the features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


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


def mnist_data() -> tuple[NumArray, NumArray, IntArray, IntArray]:
    # Load and preprocess the data
    X_train: NumArray; X_test: NumArray
    y_train: IntArray; y_test: IntArray
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(60000, 784).astype("float32") / 255
    X_test = X_test.reshape(10000, 784).astype("float32") / 255

    return X_train, X_test, y_train, y_test


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

class TextCleaner:
    lemmatizer: WordNetLemmatizer
    stop_words: set

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text) -> str:
        # Removing non-word characters
        text = re.sub(r'\W', ' ', text).lower()

        cleaned_text: list[str] = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ]

        return ' '.join(cleaned_text)


def spam_data(testing_proportion: float = 0.2,
              vocabulary_size: int = 1000,
              max_sequence_length: int = 50) -> tuple[NumArray, NumArray,
                                                      IntArray, IntArray]:
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
    cleaner: TextCleaner = TextCleaner()
    Data['text'] = Data['text'].apply(cleaner.clean)

    X_train: NumArray; X_test: NumArray
    y_train: IntArray; y_test: IntArray

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

    X_train_padded: NumArray = pad_sequences(
        X_train_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )
    X_test_padded: NumArray = pad_sequences(
        X_test_seq,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post'
    )

    return X_train_padded, X_test_padded, y_train, y_test


def get_dataset(params: dict) -> tuple[NumArray, NumArray, IntArray, IntArray]:
    if as_name(params['dataset']) == "mnist":
        return mnist_data()
    elif as_name(params['dataset']) == "sms_spam":
        return spam_data(
            testing_proportion=params['testing_proportion'],
            vocabulary_size=params['vocabulary_size'],
            max_sequence_length=params['max_sequence_length'],
        )
    else:
        raise ValueError(f"{params['dataset']} is not a valid dataset")


def get_model_by_dataset(params: dict) -> KerasModel:
    if as_name(params['dataset']) == "mnist":
        return mnist_model(
            learning_rate=params['learning_rate'],
            dense_units=params['dense_units'],
        )
    elif as_name(params['dataset']) == "sms_spam":
        return spam_model(
            learning_rate=params['learning_rate'],
            vocabulary_size=params['vocabulary_size'],
            sequence_length=params['max_sequence_length'],
            embedding_dim=params['embedding_dim'],
            lstm_units=params['lstm_units'],
        )
    else:
        raise ValueError(f"{params['dataset']} is not a valid dataset")


def get_controller_by_protocol(params: dict,
                               nodes: list[Node],
                               global_model: KerasModel,
                               x_testing: NumArray,
                               y_testing: NumArray,
                               metrics_params: dict[str,
                                                    MetricParams],
                               attacker: Union[Attacker,
                                               None] = None) -> Learning:
    base_params: dict = {
        'iterations': params['iterations_num'],
        'nodes': nodes,
        'global_model': global_model,
        'x_testing': x_testing,
        'y_testing': y_testing,
        'metrics_params': metrics_params,
        'attacker': attacker,
    }

    protocol: str = as_name(params['protocol'])

    if protocol in ["dl", "baseline"]:
        return FederatedLearning(
            weighting_mode=params['weighting_mode'],
            **base_params
        )
    elif protocol == "sydelp":
        return Sydelp(
            weighting_mode=params['weighting_mode'],
            expected_malicious_num=params['expected_malicious_num'],
            **base_params
        )
    elif protocol == "mab-fl":
        return Mab(
            warm_up_iterations=params['warm_up_iterations'],
            alpha=params['mab_alpha'],
            miu=params['miu'],
            c_max=params['c_max'],
            c_min=params['c_min'],
            pca_components=params['pca_components'],
            **base_params
        )
    else:
        raise ValueError(f'Protocol "{params["protocol"]}" not recognized')


def get_metrics(params: dict) -> dict[str, MetricParams]:
    metrics: dict[str, MetricParams] = {}

    for metric in params['metrics']:
        if metric == 'accuracy':
            metrics['accuracy'] = {'function': accuracy_score, 'params': {}}
        elif metric == 'f1_score':
            metrics['f1_score'] = {'function': f1_score, 'params': {}}
        elif metric == 'attack_success_rate':
            metrics['attack_success_rate'] = {
                'function': label_flipping_success_rate,
                'params': {
                    'source': params['source_label'],
                    'target': params['target_label']
                }
            }
        elif metric == 'label_recall':
            metrics['label_recall'] = {
                'function': label_recall,
                'params': {'label': params['source_label']}
            }
        else:
            raise ValueError(f'metric "{metric}" not recognized')

    return metrics


def get_nodes_by_protocol(params: dict,
                          X_train: NumArray,
                          y_train: IntArray,
                          models: list[KerasModel]) -> list[Node]:
    if params['nodes_num'] != len(models):
        raise ValueError("params['nodes_num'] != len(models)")

    base_node_params: dict = {
        'epochs': params['local_epochs_num'],
        'batch_size': params['batch_size']
    }

    NodeClass = Node
    RandomClass = RandomNode
    SignClass = SignFlippingNode
    TarLabelClass = TargetedLabelFlippingNode

    if as_name(params['protocol']) == "mab-fl":
        NodeClass = MabNode
        RandomClass = MabRandomNode
        SignClass = MabSignFlippingNode
        TarLabelClass = MabTargetedLabelFlippingNode

    malicious_num: int = (
        params.get('random_malicious_num', 0)
        + params.get('sign_flip_malicious_num', 0)
        + params.get('label_flip_malicious_num', 0)
    )
    honest_num: int = params['nodes_num'] - malicious_num

    nodes: list[Node] = []

    if malicious_num > 0:
        # Divide the training set for malicious users proportionally
        X_train, X_mal, y_train, y_mal = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            test_size=malicious_num/params['nodes_num']
        )

        mal_splits: list[Split] = balanced_split(
            X_mal,
            y_mal,
            n_splits=malicious_num,
            seed=params['seed'],
        )

        for _ in range(params.get('random_malicious_num', 0)):
            split: Split = mal_splits.pop()

            nodes.append(RandomClass(
                mean=params['attack_mean'],
                sd=params['attack_sd'],

                x=split['X'],
                y=split['y'],
                model=models.pop(),
                **base_node_params
            ))

        for _ in range(params.get('sign_flip_malicious_num', 0)):
            split: Split = mal_splits.pop()

            nodes.append(SignClass(
                scale_factor=params['attack_scale_factor'],

                x=split['X'],
                y=split['y'],
                model=models.pop(),
                **base_node_params
            ))

        for _ in range(params.get('label_flip_malicious_num', 0)):
            split: Split = mal_splits.pop()

            nodes.append(TarLabelClass(
                source=params['source_label'],
                target=params['target_label'],

                x=split['X'],
                y=split['y'],
                model=models.pop(),
                **base_node_params
            ))

    honest_splits: list[Split] = dirichlet_split(
        X_train,
        y_train,
        n_splits=honest_num,
        alpha=params['alpha'],
        split_min_size=params['split_min_size'],
        seed=params['seed'],
    )

    for _ in range(honest_num):
        split: Split = honest_splits.pop()

        nodes.append(NodeClass(x=split['X'],
                               y=split['y'],
                               model=models.pop(),
                               **base_node_params))

    return nodes


def get_attacker(nodes: list[Node], params: dict) -> Union[Attacker, None]:
    attack: str = as_name(params['attack'])

    if attack in ['random', 'solitary_untargeted']:
        AttackerClass = RandomAttacker
        NodeClass = RandomNode
    elif attack == 'sign_flipping':
        AttackerClass = SignFlippingAttacker
        NodeClass = SignFlippingNode
    elif attack in ['label_flipping', 'solitary_targeted']:
        AttackerClass = TargetedLabelFlippingAttacker
        NodeClass = TargetedLabelFlippingNode
    elif 'untargeted_label_flipping':
        AttackerClass = UntargetedLabelFlippingAttacker
        NodeClass = UntargetedLabelFlippingNode
    else:
        raise ValueError(f"{params['attack']} is not a valid attack")

    class_nodes = [node for node in nodes if isinstance(node, NodeClass)]
    mal_nodes = [node for node in nodes if node.is_malicious]

    if len(class_nodes) != len(mal_nodes):
        raise ValueError("len(class_nodes) != len(mal_nodes)")

    if not class_nodes:
        return None

    return AttackerClass(
        nodes=class_nodes,  # type: ignore
        is_identical_attack=params['is_identical_attack']
    )
