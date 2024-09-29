import dataclasses
import enum
from collections.abc import Callable
from typing import Any, Protocol

import numpy
import sklearn.ensemble
import xgboost


@enum.unique
class Problem(enum.Enum):
    """A kind of prediction problem."""

    classification = enum.auto()
    regression = enum.auto()


class Model(Protocol):
    """A supervised model that conforms to the sklearn estimator interface."""

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
    ) -> Any:
        ...

    def predict(
        self,
        X: numpy.ndarray,
    ) -> numpy.ndarray:
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class Solver:
    """A model constructor for a particular kind of prediction problem.

    :param factory: Model constructor.
    :param name: Name of the solver.
    :param problem: Type of problem.
    """

    factory: Callable[[], Model]
    name: str
    problem: Problem


SOLVERS = [
    Solver(
        factory=xgboost.XGBClassifier,
        name="XGBoost",
        problem=Problem.classification,
    ),
    Solver(
        factory=xgboost.XGBRegressor,
        name="XGBoost",
        problem=Problem.regression,
    ),
    Solver(
        factory=sklearn.ensemble.RandomForestClassifier,
        name="Random Forest",
        problem=Problem.classification,
    ),
    Solver(
        factory=sklearn.ensemble.RandomForestRegressor,
        name="Random Forest",
        problem=Problem.regression,
    ),
]
