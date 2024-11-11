import dataclasses
from collections.abc import Collection

import numpy
import sklearn.feature_selection

from autofeat.problem import Problem
from autofeat.selector.base import Selector


@dataclasses.dataclass(kw_only=True)
class MutualInformation(Selector):
    """Select the top ``n`` features by mutual information with the target variable.

    :param n: Number of features to select.
    :param problem: Type of prediction problem.
    """

    n: int
    problem: Problem

    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        if self.n >= X.shape[1]:
            return [True] * X.shape[1]

        scorer = (
            sklearn.feature_selection.mutual_info_classif
            if self.problem == Problem.classification
            else sklearn.feature_selection.mutual_info_regression
        )

        selector = sklearn.feature_selection.SelectKBest(scorer, k=self.n)

        selector.fit(numpy.nan_to_num(X), numpy.nan_to_num(y))

        return selector.get_support()  # type: ignore[no-any-return]
