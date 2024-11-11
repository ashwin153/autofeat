import dataclasses

import numpy
import sklearn.inspection
import sklearn.utils

from autofeat.predictor.base import Predictor
from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class PermutationImportance(Selector):
    """Select the ``n`` most important features to the ``predictor`` through permutation analysis.

    :param n: Number of features to select.
    :param predictor: Prediction model.
    """

    n: int
    predictor: Predictor

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> list[bool]:
        if self.n >= X.shape[1]:
            return [True] * X.shape[1]

        self.predictor.fit(X, y)

        result = sklearn.inspection.permutation_importance(
            self.predictor,
            X, y,
        )

        assert isinstance(result, sklearn.utils.Bunch)

        selection = (
            numpy
            .subtract(result.importances_mean, result.importances_std * 2)
            .argpartition(-self.n)[-self.n:]
        )

        return [
            i in selection
            for i in range(X.shape[1])
        ]
