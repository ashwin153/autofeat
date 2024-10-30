import dataclasses
from collections.abc import Collection

import boruta
import numpy

from autofeat.predictor.base import Predictor
from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class Boruta(Selector):
    """Select the most important features to the ``predictor`` using the Boruta algorithm.

    :param predictor: Prediction model.
    """

    predictor: Predictor

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        selector = boruta.BorutaPy(
            estimator=self.predictor,
            n_estimators="auto",  # pyright: ignore[reportArgumentType]
            perc=100,
            alpha=0.05,
            max_iter=100,
        )

        selector.fit(X, y)

        return selector.support_  # type: ignore[no-any-return]
