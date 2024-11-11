import dataclasses
from typing import Any, assert_never

import lightgbm

from autofeat.predict.base import PredictionMethod, Predictor
from autofeat.problem import Problem
from autofeat.settings import SETTINGS


@dataclasses.dataclass(frozen=True, kw_only=True)
class LightGBM(PredictionMethod):
    """An ensemble of gradient boosted decision trees."""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        parameters: dict[str, Any] = {
            "device": "cuda" if SETTINGS.polars_engine == "gpu" else "cpu",
        }

        match problem:
            case Problem.classification:
                return lightgbm.LGBMClassifier(**parameters)  # pyright: ignore[reportReturnType]
            case Problem.regression:
                return lightgbm.LGBMRegressor(**parameters)  # pyright: ignore[reportReturnType]
            case _:
                assert_never(problem)
