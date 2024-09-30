import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, Protocol

import catboost
import lightgbm
import numpy
import polars
import shap
import sklearn.ensemble
import sklearn.metrics
import xgboost


@enum.unique
class Problem(enum.Enum):
    """A kind of prediction problem."""

    classification = enum.auto()
    regression = enum.auto()

    def __str__(
        self,
    ) -> str:
        return self.name


class Model(Protocol):
    """A supervised model that conforms to the sklearn estimator interface."""

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Solver:
    """A constructor for a model that solves a particular kind of prediction problem.

    :param factory: Model constructor.
    :param name: Name of the solver.
    :param problem: Type of problem.
    """

    factory: Callable[[], Model]
    name: str
    problem: Problem

    def __str__(
        self,
    ) -> str:
        return self.name


SOLVERS = [
    Solver(
        factory=catboost.CatBoostClassifier,
        name="CatBoost",
        problem=Problem.classification,
    ),
    Solver(
        factory=catboost.CatBoostRegressor,
        name="CatBoost",
        problem=Problem.regression,
    ),
    Solver(
        factory=lightgbm.LGBMClassifier,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
        problem=Problem.classification,
    ),
    Solver(
        factory=lightgbm.LGBMRegressor,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
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
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Solution:
    """A solution to a prediction problem.

    :param model: Trained model.
    :param problem: Type of problem.
    :param X_test: Input variables used to test the ``model``.
    :param X_train: Input variables used to train the ``model``.
    :param y_test: Target variable used to test the ``model``.
    :param y_train: Target variable used to train the ``model``.
    :param y_pred: Target variable that was predicted by the ``model`` in the test.
    """

    model: Model
    problem: Problem
    X_test: polars.DataFrame
    X_train: polars.DataFrame
    y_test: polars.Series
    y_train: polars.Series
    y_pred: polars.Series

    @functools.cached_property
    def shap_values(  # type: ignore[no-any-unimported]
        self,
    ) -> shap.Explanation:
        """Get the SHAP values associated with the model.

        :return: SHAP values.
        """
        explainer = shap.Explainer(
            model=self.model,
            feature_names=self.X_train.columns,
        )

        return explainer(self.X_test.to_numpy())
