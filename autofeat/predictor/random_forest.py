from typing import assert_never

import catboost

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


class RandomForest(PredictionMethod):
    """"""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return catboost.CatBoostClassifier()
            case Problem.regression:
                return catboost.CatBoostRegressor()
            case _:
                assert_never(problem)
