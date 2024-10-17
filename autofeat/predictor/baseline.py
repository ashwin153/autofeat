from typing import assert_never

import sklearn.dummy

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


class Baseline(PredictionMethod):
    """Baseline prediction method to benchmark the performance of other prediction methods."""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return sklearn.dummy.DummyClassifier(strategy="most_frequent")
            case Problem.regression:
                return sklearn.dummy.DummyRegressor(strategy="mean")  # pyright: ignore[reportReturnType]
            case _:
                assert_never(problem)