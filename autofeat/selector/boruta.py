from collections.abc import Collection

import attrs
import boruta
import numpy

from autofeat.model import PredictionModel
from autofeat.selector.base import Selector


@attrs.define(frozen=True, kw_only=True)
class Boruta(Selector):
    """Select the most important features to the ``predictor`` using the Boruta algorithm.

    :param predictor: Prediction model.
    """

    predictor: PredictionModel

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        selector = boruta.BorutaPy(self.predictor, n_estimators="auto")

        selector.fit(numpy.nan_to_num(X), numpy.nan_to_num(y))

        return selector.support_  # type: ignore[no-any-return]
