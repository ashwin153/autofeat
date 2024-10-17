import boruta
import numpy

from autofeat.model import Predictor
from autofeat.selector.base import Selector


class Boruta(Selector):
    """Select the most important features to the ``predictor`` using the Boruta algorithm.

    :param predictor: Prediction model.
    """

    predictor: Predictor

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> list[bool]:
        selector = boruta.BorutaPy(self.predictor, n_estimators="auto")
        selector.fit(numpy.nan_to_num(X), numpy.nan_to_num(y))
        return selector.support_.tolist()
