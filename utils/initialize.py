import networkx as nx
import os

from random import shuffle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Union
from utils.typing import NumArray, IntArray, KerasModel, WeightsShapes
from utils.utils import as_name
from utils.split import Split, balanced_split, dirichlet_split
from utils.metrics import f1_score, label_flipping_success_rate, label_recall
from utils.graph import create_random_geometric_graph

from learning.learning import MetricParams, Learning
from learning.federated import FederatedLearning
from learning.sydelp import Sydelp
from learning.mab import Mab
from learning.sybilwall import Sybilwall
from learning.gossip import Gossip

from attack.attacker import Attacker
from attack.sydelp_attacker import SydelpAttacker

from nodes.node import Node
from nodes.random_node import RandomNode
from nodes.targeted_label_flipping_node import TargetedLabelFlippingNode
from nodes.sign_flipping_node import SignFlippingNode
from nodes.mab_malicious_nodes import *
from nodes.sybilwall_malicious_nodes import *
from nodes.mab_node import MabNode
from nodes.sybilwall_node import SybilwallNode
from nodes.sydelp_node import SydelpNode
from nodes.sydelp_malicious_nodes import *

from utils.datasets import mnist_data, spam_data
from utils.models import mnist_model, spam_model


def get_dataset(params: dict) -> tuple[NumArray, NumArray, IntArray, IntArray]:
    if as_name(params['dataset']) == "mnist":
        return mnist_data()
    elif as_name(params['dataset']) == "sms_spam":
        return spam_data(
            testing_proportion=params['testing_proportion'],
            vocabulary_size=params['vocabulary_size'],
            max_sequence_length=params['max_sequence_length']
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
    weights_shapes: WeightsShapes = [layer.shape
                                     for layer in global_model.get_weights()]
    honest_nodes: list[Node] = [node for node in nodes if not node.is_malicious]

    base_params: dict = {
        'iterations_num': params['iterations_num'],
        'honest_nodes': honest_nodes,
        'global_model': global_model,
        'weights_shapes': weights_shapes,
        'x_testing': x_testing,
        'y_testing': y_testing,
        'metrics_params': metrics_params,
        'attacker': attacker,
    }

    protocol: str = as_name(params['protocol'])

    if protocol == "fl":
        return FederatedLearning(**base_params)
    elif protocol == "sydelp":
        return Sydelp(
            expected_malicious_num=params['expected_malicious_num'],
            momentum_coeff=params['momentum_coeff'],
            difficulty_alpha=params['difficulty_alpha'],
            models_per_iteration=params['models_per_iteration'],
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
    elif protocol == "gossip":
        graph: nx.Graph = nx.random_regular_graph(
            d=params['degree'],
            n=params['nodes_num'],
            seed=params['seed'],
        )

        return Gossip(graph=graph, **base_params)
    elif protocol == "sybilwall":
        graph_type: str = as_name(params['graph_type'])

        if graph_type == "random_geometric":
            graph: nx.Graph = create_random_geometric_graph(
                nodes_num=params['nodes_num'],
                max_degree=params['max_degree'],
                edge_prob=params['edge_prob'],
                seed=params['seed'],
            )
        elif graph_type == "random_regular":
            graph: nx.Graph = nx.random_regular_graph(
                d=params['degree'],
                n=params['nodes_num'],
                seed=params['seed'],
            )
        else:
            raise ValueError('Graph type %s not recognized' %
                             params["graph_type"])

        if not nx.is_connected(graph):
            raise ValueError("The generated graph is not connected")

        return Sybilwall(graph=graph, **base_params)
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

    weights_shapes: WeightsShapes = [layer.shape
                                     for layer in models[0].get_weights()]
    protocol = as_name(params['protocol'])
    attack = as_name(params['attack'])
    base_node_params: dict = {
        'weights_shapes': weights_shapes,
        'epochs': params['local_epochs_num'],
        'batch_size': params['batch_size']
    }

    NodeClass = Node
    RandomClass = RandomNode
    SignClass = SignFlippingNode
    TarLabelClass = TargetedLabelFlippingNode

    if protocol == "mab-fl":
        NodeClass = MabNode
        RandomClass = MabRandomNode
        SignClass = MabSignFlippingNode
        TarLabelClass = MabTargetedLabelFlippingNode
    elif protocol == "sydelp":
        base_node_params['momentum_coeff'] = params['momentum_coeff']
        base_node_params['difficulty_alpha'] = params['difficulty_alpha']
        base_node_params['iterations_num'] = params['iterations_num']

        NodeClass = SydelpNode
        RandomClass = SydelpRandomNode
        SignClass = SydelpSignFlippingNode
        TarLabelClass = SydelpTargetedLabelFlippingNode
    elif protocol == "sybilwall":
        base_node_params['confidence'] = params['confidence']
        base_node_params['distant_propagation_relevance'] = params['distant_propagation_relevance']

        NodeClass = SybilwallNode
        RandomClass = SybilwallRandomNode
        SignClass = SybilwallSignFlippingNode
        TarLabelClass = SybilwallTargetedLabelFlippingNode
    elif protocol not in ["fl", "sydelp", "gossip"]:
        raise ValueError(f'Protocol "{params["protocol"]}" not recognized')

    malicious_num: int = (
        params.get('random_malicious_num', 0)
        + params.get('sign_flip_malicious_num', 0)
        + params.get('label_flip_malicious_num', 0)
    )
    honest_num: int = params['nodes_num'] - malicious_num

    nodes: list[Node] = []

    if malicious_num > 0:
        if (attack in ["sign_flipping", "label_flipping"]
            or (protocol == "sydelp" and params['is_worst_case'])):
            # Divide the training set for malicious users proportionally
            X_train, X_mal, y_train, y_mal = train_test_split(
                X_train,
                y_train,
                stratify=y_train,
                test_size=min(malicious_num/params['nodes_num'], 0.15),
            )

            mal_splits: list[Split] = balanced_split(
                X_mal,
                y_mal,
                n_splits=malicious_num,
                seed=params['seed'],
            )
        else:
            # In this cases, datasets are not required
            mal_splits: list[Split] = [{'X': array([]), 'y': array([])}
                                       for _ in range(malicious_num)]

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

    # Non-IID partitioning for honest updates
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

    if protocol == "sybilwall":
        shuffle(nodes)

    return nodes


def get_attacker(nodes: list[Node], params: dict) -> Union[Attacker, None]:
    mal_nodes: list[Node] = [node for node in nodes if node.is_malicious]

    if not mal_nodes:
        return None

    # For sydelp worst case attack use the same value for the number of
    # malicious nodes and the computing power for proper initialization.
    # Otherwise, the necessary malicious nodes will be included in the second
    # iteration.
    if as_name(params['protocol']) == "sydelp":
        return SydelpAttacker(
            nodes=mal_nodes,  # type: ignore
            is_identical_attack=params['is_identical_attack'],

            computing_power=params['computing_power'],
            is_worst_case=params['is_worst_case'],
            objective_malicious_num=params['expected_malicious_num'] + 1,
        )
    else:
        return Attacker(
            nodes=mal_nodes, # type: ignore
            is_identical_attack=params['is_identical_attack']
        )


def get_results_dir(params: dict) -> str:
    additional: str = ""

    if as_name(params['protocol']) == "sydelp":
        additional = os.path.join(as_name(params['is_worst_case']),
                                  f"alpha_{params['difficulty_alpha']}",
                                  as_name(params['expected_malicious_num']),
                                  as_name(params['models_per_iteration']),
                                  f"P_{params['computing_power']}")

    return os.path.join(
        params['results_dir'],
        as_name(params['protocol']),
        as_name(params['dataset']),
        as_name(params['attack']),
        as_name(params['seed']),
        as_name(params.get('is_identical_attack', 'no_attack')),
        additional
    )

def get_metadata(params: dict) -> dict:
    metadata = {
        'Protocol': params['protocol'],
        'Dataset': params['dataset'],
        'Attack': params['attack'],
        'Seed': params['seed'],
        'IdenticalAttack': as_name(params.get('is_identical_attack',
                                              'no_attack')),
    }

    if as_name(params['protocol']) == "sydelp":
        metadata['WorstCase'] = params['is_worst_case']
        metadata['DifficultyAlpha'] = params['difficulty_alpha']
        metadata['ExpectedMaliciousNum'] = params['expected_malicious_num']
        metadata['ModelsPerIteration'] = params['models_per_iteration']
        metadata['ComputingPower'] = params['computing_power']

    return metadata
