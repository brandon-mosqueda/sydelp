import numpy as np
import pandas as pd

from numpy.typing import NDArray
from pandas import DataFrame
from node import Node
from time import time
from utils import progress_bar, MyProgressBar, NNumeric
from keras.src.models import Model as KerasModel
from abc import ABC, abstractmethod


class CollaborativeLearning(ABC):
    nodes: list[Node]
    global_model: KerasModel
    x_testing: NDArray[NNumeric]
    y_testing: NDArray[NNumeric]
    rounds: int
    metrics: DataFrame = DataFrame()
    execution_time: float = 0

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
    def evaluate(self) -> dict:
        pass

    def start(self) -> DataFrame:
        start: float = time()
        metrics: list[dict] = []

        for i in range(self.rounds):
            print(f'* Round {i + 1}/{self.rounds}')

            bar: MyProgressBar = progress_bar(len(self.nodes))

            for node in self.nodes:
                node.train()
                bar.next()

            bar.finish()
            self.aggregation()

            round_metrics: dict = self.evaluate()
            round_metrics['round'] = i + 1
            round_metrics['time'] = time() - start

            metrics.append(round_metrics)

        self.metrics = pd.DataFrame(metrics)

        # Set the execution time in minutes
        self.execution_time = (time() - start) / 60

        return self.metrics
