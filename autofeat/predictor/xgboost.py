import dataclasses
from typing import assert_never

import xgboost

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


@dataclasses.dataclass(frozen=True, kw_only=True)
class XGBoost(PredictionMethod):
    """"""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return xgboost.XGBClassifier(device="cuda")
            case Problem.regression:
                return xgboost.XGBRegressor(device="cuda")
            case _:
                assert_never(problem)
