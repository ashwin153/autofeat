import dataclasses
from typing import Any, assert_never

import xgboost

from autofeat.predict.base import PredictionMethod, Predictor
from autofeat.problem import Problem
from autofeat.settings import SETTINGS


@dataclasses.dataclass(frozen=True, kw_only=True)
class XGBoost(PredictionMethod):
    """An ensemble of gradient boosted decision trees."""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        parameters: dict[str, Any] = {
            "device": "cuda" if SETTINGS.polars_engine == "gpu" else None,
        }

        match problem:
            case Problem.classification:
                return xgboost.XGBClassifier(**parameters)
            case Problem.regression:
                return xgboost.XGBRegressor(**parameters)
            case _:
                assert_never(problem)
