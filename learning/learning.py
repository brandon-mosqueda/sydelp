import os

import utils.utils as utils
import numpy as np
import pandas as pd

from time import time
from abc import ABC, abstractmethod
from nodes.node import Node
from attack.attacker import Attacker
from utils.utils import MyProgressBar
from pandas import DataFrame
from utils.typing import NumArray, Float, IntArray, KerasModel, WeightsShapes
from typing import TypedDict, Protocol, TypeVar, Generic, Union


class MetricFunction(Protocol):
    def __call__(self, y_true: IntArray, y_pred: IntArray,
                 *args, **kwargs) -> Float: ...


class MetricParams(TypedDict):
    function: MetricFunction
    params: dict


# This allows to use any kind of node in the learning sub-classes
NodeType = TypeVar('NodeType', bound='Node')


class Learning(ABC, Generic[NodeType]):
    nodes: list[NodeType]
    honest_num: int
    global_model: KerasModel
    weights_shapes: WeightsShapes
    x_testing: NumArray
    y_testing: IntArray
    iterations: int
    execution_time: float
    metrics: DataFrame
    metrics_params: dict[str, MetricParams]
    attacker: Union[Attacker, None]

    def __init__(self,
                 iterations: int,
                 nodes: list[NodeType],
                 global_model: KerasModel,
                 weights_shapes: WeightsShapes,
                 x_testing: NumArray,
                 y_testing: IntArray,
                 metrics_params: dict[str, MetricParams],
                 attacker: Union[Attacker, None] = None) -> None:
        self.nodes = nodes
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.global_model = global_model
        self.weights_shapes = weights_shapes
        self.iterations = iterations
        self.metrics_params = metrics_params
        self.execution_time = 0
        self.honest_num = sum(not node.is_malicious for node in self.nodes)
        self.attacker = attacker

    def iteration_setup(self, iteration_num: int) -> None:
        pass

    def training(self) -> None:
        bar: MyProgressBar = utils.progress_bar(self.honest_num)

        for node in self.nodes:
            if not node.is_malicious:
                node.train()
                bar.next()

        bar.finish()

    @abstractmethod
    def aggregation(self, iteration_num: int) -> None:
        pass

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

    def round_metrics(self) -> dict[str, Float]:
        predictions: DataFrame = self.round_predictions()

        loss: float = self.global_model.evaluate(self.x_testing,
                                                 self.y_testing,
                                                 verbose=0)

        round_metrics: dict[str, Float] = {
            metric: self.metrics_params[metric]['function'](
                y_true=predictions['observed'].to_numpy().astype("int"),
                y_pred=predictions['predicted'].to_numpy().astype("int"),
                **self.metrics_params[metric]['params']
            ) for metric in self.metrics_params
        }

        round_metrics['loss'] = loss

        return round_metrics

    def start(self) -> DataFrame:
        start: float = time()
        metrics: list[dict] = []

        for i in range(self.iterations):
            print(f'* Iteration {i + 1}/{self.iterations}')

            self.iteration_setup(i)

            print("\t+ Training")
            self.training()

            if self.attacker:
                print("\t+ Attacking")
                self.attacker.attack()

            self.aggregation(i)

            print("\t+ Metrics:")
            metrics_i: dict[str, Float] = self.round_metrics()
            [print("\t\t- %s: %.4f" % item) for item in metrics_i.items()]

            metrics_i['round'] = i
            metrics_i['time'] = utils.elapsed_time(start, time())

            metrics.append(metrics_i)

        self.metrics = pd.DataFrame(metrics)

        # Set the execution time in minutes
        self.execution_time = utils.elapsed_time(start, time())

        return self.metrics

    def save(self, dir: str, all: bool = False, metadata: dict = {}) -> None:
        os.makedirs(dir, exist_ok=True)

        for key in metadata:
            self.metrics[key] = metadata[key]

        self.metrics.to_csv(os.path.join(dir, "metrics.csv"), index=False)

        if all:
            self.global_model.save(os.path.join(dir, "global_model.keras"))
            os.makedirs(os.path.join(dir, "nodes"), exist_ok=True)

            for i, node in enumerate(self.nodes):
                node.model.save(os.path.join(dir, "nodes", f"node_{i}.keras"))
