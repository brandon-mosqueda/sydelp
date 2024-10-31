import os

import numpy as np
import pandas as pd
import utils.utils as utils

from typing import Callable
from numpy.typing import NDArray
from pandas import DataFrame, concat
from nodes.node import Node
from time import time
from utils.utils import MyProgressBar, NNumeric, Weights
from keras.src.models import Model as KerasModel
from utils.aggregation import fed_avg


AggFunct = Callable[[list[Weights]], Weights]

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

    aggregation_function: AggFunct
    aggregation_params: dict

    def __init__(self,
                 rounds: int,
                 nodes: list[Node],
                 global_model: type,
                 x_testing: NDArray[NNumeric],
                 y_testing: NDArray[NNumeric],
                 aggregation_function: AggFunct = fed_avg,
                 aggregation_params: dict = {}) -> None:
        self.nodes = nodes
        self.x_testing = x_testing
        self.y_testing = y_testing
        self.global_model = global_model
        self.rounds = rounds

        self.aggregation_function = aggregation_function
        self.aggregation_params = aggregation_params

    def aggregation(self) -> None:
        weights: list[Weights] = [node.get_weights() for node in self.nodes]
        avg_weights: Weights = self.aggregation_function(
            weights,
            **self.aggregation_params
        )

        self.global_model.set_weights(avg_weights)

        for node in self.nodes:
            node.set_weights(avg_weights)

    def round_metrics(self, round: int, start_time: float) -> dict:
        accuracy: NDArray[NNumeric]
        loss, accuracy = self.global_model.evaluate(self.x_testing,
                                                    self.y_testing,
                                                    verbose=0)

        metrics = {
            'round': round,
            'time': utils.elapsed_time(start_time, time()),
            'accuracy': accuracy,
            'loss': loss
        }

        return metrics

    def round_predictions(self, round: int) -> DataFrame:
        probs: DataFrame = DataFrame(self.global_model.predict(
            self.x_testing,
            verbose=0
        ))

        data: DataFrame = DataFrame({
            "round": round,
            "observed": self.y_testing,
            "predicted": np.argmax(probs, axis=1)
        })

        return concat([data, probs], axis=1)

    def round_attack_metrics(self, round: int, start_time: float) -> dict:
        return {}

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

            metrics_i: dict = self.round_metrics(i + 1, start)
            attack_metrics_i: dict = self.round_attack_metrics(i + 1, start)
            predictions_i: DataFrame = self.round_predictions(i + 1)

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
        self.predictions.to_csv(os.path.join(dir, "predictions.csv"),
                                index=False)

        if all:
            self.global_model.save(os.path.join(dir, "global_model.keras"))
            os.makedirs(os.path.join(dir, "nodes"), exist_ok=True)

            for i, node in enumerate(self.nodes):
                node.model.save(os.path.join(dir, "nodes", f"node_{i}.keras"))
