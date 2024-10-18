import dataclasses
from collections.abc import Collection
from typing import Literal

import BorutaShap
import numpy
import pandas

from autofeat.predictor.base import Predictor
from autofeat.problem import Problem
from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class Boruta(Selector):
    """Select the most important features to the ``predictor`` using the Boruta algorithm.

    :param importance: Metric to use to measure feature importance.
    :param predictor: Prediction model.
    :param problem: Type of prediction problem.
    :param trials: Number of iterations to run Boruta.
    """

    importance: Literal["shap", "gini", "perm"] = "shap"
    predictor: Predictor
    problem: Problem
    trials: int = 20

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        selector = BorutaShap.BorutaShap(
            classification=self.problem == Problem.classification,
            importance_measure=self.importance,
            model=self.predictor,
            percentile=100,
            pvalue=0.05,
        )

        selector.fit(
            pandas.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
            y,
            n_trials=self.trials,
        )

        selection = {
            int(column)
            for column in selector.accepted + selector.tentative
        }

        return [
            i in selection
            for i in range(X.shape[1])
        ]
