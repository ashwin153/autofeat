from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy

    from autofeat.problem import Problem


class Predictor(Protocol):
    """Any sklearn predictor."""

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        /,
    ) -> Any:
        ...

    def predict(
        self,
        X: numpy.ndarray,
        /,
    ) -> numpy.ndarray:
        ...


class PredictionMethod(abc.ABC):
    """A predictor factory."""

    @abc.abstractmethod
    def create(
        self,
        problem: Problem,
    ) -> Predictor:
        """Create a predictor that can solve the ``problem``.

        :param problem: Type of prediction problem.
        :return: Predictor for the problem.
        """

    @classmethod
    def __init_subclass__(
        cls,
    ) -> None:
        PREDICTION_METHODS[cls.__name__] = cls()


# TODO: should be Type[PredictionMethod] eventually
PREDICTION_METHODS: dict[str, PredictionMethod] = {}
