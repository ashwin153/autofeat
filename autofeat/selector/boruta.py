import dataclasses
import math
from collections.abc import Collection

import boruta
import numpy

from autofeat.predictor.base import Predictor
from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class Boruta(Selector):
    """Select the most important features to the ``predictor`` using the Boruta algorithm.

    :param max_iterations: Maximum number of iterations to run.
    :param percentile: Threshold below which features are considered irrelevant.
    :param predictor: Prediction model.
    :param p_value: Threshold below which results are considered statistically significant.
    """

    max_iterations: int = 25
    percentile: int = 95
    predictor: Predictor
    p_value: float = 0.05

    def __post_init__(
        self,
    ) -> None:
        assert 0 < self.max_iterations
        assert 0 < self.p_value <= 1
        assert 0 < self.percentile <= 100

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        selector = boruta.BorutaPy(
            alpha=self.p_value,
            early_stopping=True,
            estimator=self.predictor,
            max_iter=self.max_iterations,
            n_estimators="auto",  # pyright: ignore[reportArgumentType]
            n_iter_no_change=math.ceil(self.max_iterations / 4),
            perc=self.percentile,
        )

        selector.fit(numpy.nan_to_num(X), y)

        return selector.support_  # type: ignore[no-any-return]
