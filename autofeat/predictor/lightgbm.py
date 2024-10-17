from typing import assert_never

import lightgbm

from autofeat.predictor.base import PredictionMethod, Predictor
from autofeat.problem import Problem


class LightGBM(PredictionMethod):
    """"""

    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        match problem:
            case Problem.classification:
                return lightgbm.LGBMClassifier()  # pyright: ignore[reportReturnType]
            case Problem.regression:
                return lightgbm.LGBMRegressor()  # pyright: ignore[reportReturnType]
            case _:
                assert_never(problem)
