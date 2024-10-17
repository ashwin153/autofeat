import abc
from collections.abc import Collection
from typing import Any

import numpy
import sklearn.base
import sklearn.feature_selection


class Selector(
    abc.ABC,
    sklearn.base.BaseEstimator,  # type: ignore[no-any-unimported]
    sklearn.feature_selection.SelectorMixin,  # type: ignore[no-any-unimported]
):
    """A feature selection algorithm."""

    def __init__(
        self,
    ) -> None:
        self._support_mask: numpy.ndarray | None = None

    @abc.abstractmethod
    def select(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Collection[bool]:
        """Select a subset of the columns in ``X`` according to some criteria.

        :param X: Input variables. (nxm)
        :param y: Target variable. (n)
        :return: Whether or not each input variable is selected. (m)
        """

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        /,
    ) -> Any:
        self._support_mask = numpy.array(self.select(X, y), dtype=bool)

    def _get_support_mask(
        self,
    ) -> numpy.ndarray:
        assert self._support_mask is not None
        return self._support_mask

    def _more_tags(
        self,
    ) -> dict[str, bool]:
        return {
            "allow_nan": True,
        }
