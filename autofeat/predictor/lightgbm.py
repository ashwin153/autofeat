import dataclasses
from typing import assert_never

import lightgbm

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


@dataclasses.dataclass(frozen=True, kw_only=True)
class LightGBM(PredictionMethod):
    """An ensemble of gradient boosted decision trees."""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return lightgbm.LGBMClassifier(device="cpu")  # pyright: ignore[reportReturnType]
            case Problem.regression:
                return lightgbm.LGBMRegressor(device="cpu")  # pyright: ignore[reportReturnType]
            case _:
                assert_never(problem)
