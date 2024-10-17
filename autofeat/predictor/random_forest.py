import dataclasses
from typing import assert_never

import catboost

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomForest(PredictionMethod):
    """"""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return catboost.CatBoostClassifier()  # type: ignore[no-any-return]
            case Problem.regression:
                return catboost.CatBoostRegressor()  # type: ignore[no-any-return]
            case _:
                assert_never(problem)
