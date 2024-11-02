import os

import numpy as np
import pandas as pd
import utils.utils as utils

from typing import Callable, TypedDict
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame, concat
from nodes.node import Node
from time import time
from utils.utils import MyProgressBar, NNumeric, Weights, Float
from keras.src.models import Model as KerasModel
from utils.aggregation import fed_avg
from sklearn.metrics import accuracy_score


class AggParams(TypedDict):
    function: Callable[[list[Weights]], Weights]
    params: dict


class MetricParams(TypedDict):
    function: Callable[[ArrayLike, ArrayLike], Float]
    params: dict


class FederatedLearning:
    nodes: list[Node]
    global_model: KerasModel
    x_testing: NDArray[NNumeric]
    y_testing: NDArray[NNumeric]
    rounds: int
    execution_time: float = 0
    metrics: DataFrame
    attack_metrics: DataFrame
    predictions: DataFrame

    aggregation_params: AggParams
    metrics_params: dict[str, MetricParams]
    attack_metrics_params: dict[str, MetricParams]

    def __init__(self,
                 rounds: int,
                 nodes: list[Node],
                 global_model: KerasModel,
                 x_testing: NDArray[NNumeric],
                 y_testing: NDArray[NNumeric],
                 aggregation_params: AggParams = {
                     'function': fed_avg, 'params': {}},
                 metrics_params: dict[str, MetricParams] = {
                     'accuracy': {'function': accuracy_score, 'params': {}}
                 },
                 attack_metrics_params: dict[str, MetricParams] = {}) -> None:
        self.nodes = nodes
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.global_model = global_model
        self.rounds = rounds

        self.aggregation_params = aggregation_params
        self.metrics_params = metrics_params
        self.attack_metrics_params = attack_metrics_params

    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = self.aggregation_params['function'](
            weights,
            **self.aggregation_params['params']
        )

        self.global_model.set_weights(avg_weights)

        for node in self.nodes:
            node.set_weights(avg_weights)

    def round_predictions(self) -> DataFrame:
        preds: DataFrame = DataFrame(self.global_model.predict(
            self.x_testing,
            verbose=0
        ))

        # For binary classification
        if preds.shape[1] == 1:
            preds = pd.concat([1 - preds[0], preds], axis=1)

        preds['predicted'] = np.argmax(preds, axis=1)
        preds['observed'] = self.y_testing

        return preds

    def round_metrics(self, predictions: DataFrame) -> dict[str, Float]:
        loss: float = self.global_model.evaluate(self.x_testing,
                                                 self.y_testing,
                                                 verbose=0)

        round_metrics = {
            metric: self.metrics_params[metric]['function'](
                predictions['observed'],
                predictions['predicted'],
                **self.metrics_params[metric]['params']
            ) for metric in self.metrics_params
        }

        round_metrics['loss'] = loss

        return round_metrics

    def round_attack_metrics(self, predictions: DataFrame) -> dict[str, Float]:
        round_metrics = {
            metric: self.attack_metrics_params[metric]['function'](
                predictions['observed'],
                predictions['predicted'],
                **self.attack_metrics_params[metric]['params']
            ) for metric in self.attack_metrics_params
        }

        return round_metrics

    def start(self) -> DataFrame:
        start: float = time()
        metrics: list[dict] = []
        attack_metrics: list[dict] = []
        predictions: list[DataFrame] = []

        for i in range(self.rounds):
            print(f'* Round {i + 1}/{self.rounds}')

            bar: MyProgressBar = utils.progress_bar(len(self.nodes))

            for node in self.nodes:
                node.train()
                bar.next()

            bar.finish()
            self.aggregation()

            predictions_i: DataFrame = self.round_predictions()
            predictions_i['round'] = i

            metrics_i: dict[str, Float] = self.round_metrics(predictions_i)
            metrics_i['round'] = i
            metrics_i['time'] = utils.elapsed_time(start, time())

            if self.attack_metrics_params:
                attack_metrics_i: dict[str, Float] = self.round_attack_metrics(
                    predictions_i)
                attack_metrics_i['round'] = i
                attack_metrics_i['time'] = metrics_i['time']
            else:
                attack_metrics_i = {}

            metrics.append(metrics_i)
            attack_metrics.append(attack_metrics_i)
            predictions.append(predictions_i)

        self.attack_metrics = pd.DataFrame(attack_metrics)
        self.metrics = pd.DataFrame(metrics)
        self.predictions = pd.concat(predictions)

        # Set the execution time in minutes
        self.execution_time = utils.elapsed_time(start, time())

        return self.metrics

    def save(self, dir: str, all: bool = False) -> None:
        os.makedirs(dir, exist_ok=True)

        self.metrics.to_csv(os.path.join(dir, "metrics.csv"),
                            index=False)
        self.attack_metrics.to_csv(os.path.join(dir, "attack_metrics.csv"),
                                   index=False)
        self.predictions.to_csv(os.path.join(dir, "predictions.csv"),
                                index=False)

        if all:
            self.global_model.save(os.path.join(dir, "global_model.keras"))
            os.makedirs(os.path.join(dir, "nodes"), exist_ok=True)

            for i, node in enumerate(self.nodes):
                node.model.save(os.path.join(dir, "nodes", f"node_{i}.keras"))
