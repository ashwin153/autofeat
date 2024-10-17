from collections.abc import Collection

import attrs
import numpy
import sklearn.feature_selection

from autofeat.model import PredictionModel
from autofeat.selector.base import Selector


@attrs.define(frozen=True, kw_only=True)
class FeatureImportance(Selector):
    """Select the top ``n`` features by importance to the ``predictor``.

    :param n: Number of features to select.
    :param predictor: Prediction model.
    """

    n: int
    predictor: PredictionModel

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        if self.n >= X.shape[1]:
            return [True] * X.shape[1]

        selector = sklearn.feature_selection.SelectFromModel(
            self.predictor,
            max_features=self.n,
        )

        selector.fit(X, y)

        return selector.get_support()  # type: ignore[no-any-return]
