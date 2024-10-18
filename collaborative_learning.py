import numpy as np
import pandas as pd
import os

from numpy.typing import NDArray
from pandas import DataFrame
from nodes.node import Node
from time import time
from utils import progress_bar, MyProgressBar, NNumeric, elapsed_time
from keras.src.models import Model as KerasModel
from abc import ABC, abstractmethod


class CollaborativeLearning(ABC):
    nodes: list[Node]
    global_model: KerasModel
    x_testing: NDArray[NNumeric]
    y_testing: NDArray[NNumeric]
    rounds: int
    execution_time: float = 0
    metrics: DataFrame
    predictions: DataFrame

    def __init__(self,
                 rounds: int,
                 nodes: list[Node],
                 global_model: type,
                 x_testing: NDArray[NNumeric],
                 y_testing: NDArray[NNumeric]) -> None:
        self.nodes = nodes
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.global_model = global_model
        self.rounds = rounds

    def aggregation(self) -> None:
        pass

    @abstractmethod
    def round_metrics(self, round: int, start_time: float) -> dict:
        pass

    @abstractmethod
    def round_predictions(self, round: int) -> DataFrame:
        pass

    def start(self) -> DataFrame:
        start: float = time()
        metrics: list[dict] = []
        predictions: list[DataFrame] = []

        for i in range(self.rounds):
            print(f'* Round {i + 1}/{self.rounds}')

            bar: MyProgressBar = progress_bar(len(self.nodes))

            for node in self.nodes:
                node.train()
                bar.next()

            bar.finish()
            self.aggregation()

            metrics_i: dict = self.round_metrics(i + 1, start)
            predictions_i: DataFrame = self.round_predictions(i + 1)

            metrics.append(metrics_i)
            predictions.append(predictions_i)

        self.metrics = pd.DataFrame(metrics)
        self.predictions = pd.concat(predictions)

        # Set the execution time in minutes
        self.execution_time = elapsed_time(start, time())

        return self.metrics

    def save(self, dir: str, all: bool = False) -> None:
        os.makedirs(dir, exist_ok=True)

        self.metrics.to_csv(os.path.join(dir, "metrics.csv"),
                            index=False)
        self.predictions.to_csv(os.path.join(dir, "predictions.csv"),
                                index=False)

        if all:
            self.global_model.save(os.path.join(dir, "global_model.keras"))
            os.makedirs(os.path.join(dir, "nodes"), exist_ok=True)

            for i, node in enumerate(self.nodes):
                node.model.save(os.path.join(dir, "nodes", f"node_{i}.keras"))
