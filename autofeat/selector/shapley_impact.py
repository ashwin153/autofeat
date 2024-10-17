import attrs
import numpy
import shap

from autofeat.predictor.base import Predictor
from autofeat.selector.base import Selector


@attrs.define(frozen=True, kw_only=True)
class ShapelyImpact(Selector):
    """Select the top ``n`` features by average SHAP impact across the training data.

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

        explanation = shap.Explainer(self.predictor)(X)

        selection = (
            numpy
            .abs(explanation.values)
            .mean(tuple(i for i in range(len(explanation.shape)) if i != 1))
            .argpartition(-self.n)[-self.n:]
        )

        return [
            i in selection
            for i in range(X.shape[1])
        ]
