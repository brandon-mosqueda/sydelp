import os

import numpy as np
import pandas as pd
import utils.utils as utils

from typing import Callable, TypedDict, Protocol
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame, concat
from nodes.node import Node
from time import time
from utils.utils import MyProgressBar, NNumeric, ModelParams, Float
from keras.src.models import Model as KerasModel
from utils.aggregation import fed_avg
from sklearn.metrics import accuracy_score


class AggParams(TypedDict):
    function: Callable[[list[ModelParams]], ModelParams]
    params: dict


class MetricFunction(Protocol):
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike,
                 *args, **kwargs) -> Float: ...


class MetricParams(TypedDict):
    function: MetricFunction
    params: dict


class FederatedLearning:
    nodes: list[Node]
    global_model: KerasModel
    x_testing: NDArray[NNumeric]
    y_testing: NDArray[NNumeric]
    rounds: int
    execution_time: float = 0
    metrics: DataFrame
    predictions: DataFrame

    aggregation_params: AggParams
    metrics_params: dict[str, MetricParams]

    def __init__(self,
                 rounds: int,
                 nodes: list[Node],
                 global_model: KerasModel,
                 x_testing: NDArray[NNumeric],
                 y_testing: NDArray[NNumeric],
                 aggregation_params: AggParams = {
                     'function': fed_avg, 'params': {}
                 },
                 metrics_params: dict[str, MetricParams] = {
                     'accuracy': {'function': accuracy_score, 'params': {}}
                 }) -> None:
        self.nodes = nodes
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.global_model = global_model
        self.rounds = rounds

        self.aggregation_params = aggregation_params
        self.metrics_params = metrics_params

    def aggregation(self) -> None:
        models: list[ModelParams] = [node.get_model_params() for node in self.nodes]
        avg_models: ModelParams = self.aggregation_params['function'](
            models,
            **self.aggregation_params['params']
        )

        self.global_model.set_weights(avg_models)

        for node in self.nodes:
            node.set_model_params(avg_models)

    def round_predictions(self) -> DataFrame:
        preds: DataFrame = DataFrame(self.global_model.predict(
            self.x_testing,
            verbose=0
        ))

        # For binary classification
        if preds.shape[1] == 1:
            preds = pd.concat([1 - preds[0], preds], axis=1)
            preds.columns = [0, 1]

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

    def start(self) -> DataFrame:
        start: float = time()
        metrics: list[dict] = []
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
            print("\t- Metrics:", {
                key: round(value, 4) for key, value in metrics_i.items()
            })
            metrics_i['round'] = i
            metrics_i['time'] = utils.elapsed_time(start, time())

            metrics.append(metrics_i)
            predictions.append(predictions_i)

        self.metrics = pd.DataFrame(metrics)
        self.predictions = pd.concat(predictions)

        # Set the execution time in minutes
        self.execution_time = utils.elapsed_time(start, time())

        return self.metrics

    def save(self, dir: str, all: bool = False, metadata: dict = {}) -> None:
        os.makedirs(dir, exist_ok=True)

        for key in metadata:
            self.metrics[key] = metadata[key]
            self.predictions[key] = metadata[key]

        self.metrics.to_csv(os.path.join(dir, "metrics.csv"), index=False)

        self.predictions.to_csv(os.path.join(dir, "predictions.csv"),
                                index=False)

        if all:
            self.global_model.save(os.path.join(dir, "global_model.keras"))
            os.makedirs(os.path.join(dir, "nodes"), exist_ok=True)

            for i, node in enumerate(self.nodes):
                node.model.save(os.path.join(dir, "nodes", f"node_{i}.keras"))
