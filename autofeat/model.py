from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import boruta
import catboost
import lightgbm
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.metrics
import sklearn.model_selection
import xgboost

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy
    import polars

    from autofeat.convert import IntoDataFrame
    from autofeat.dataset import Dataset


class PredictionModel(Protocol):
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


@enum.unique
class PredictionProblem(enum.Enum):
    """A kind of prediction problem."""

    classification = enum.auto()
    regression = enum.auto()

    def __str__(
        self,
    ) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, kw_only=True)
class PredictionMethod:
    """A method of solving prediction problems.

    :param model: Model constructor.
    :param name: Name of the method.
    :param problem: Types of problems that this method can solve.
    """

    model: Callable[[], PredictionModel]
    name: str
    problem: PredictionProblem

    def __str__(
        self,
    ) -> str:
        return self.name


PREDICTION_METHODS = [
    PredictionMethod(
        model=xgboost.XGBClassifier,
        name="XGBoost",
        problem=PredictionProblem.classification,
    ),
    PredictionMethod(
        model=xgboost.XGBRegressor,
        name="XGBoost",
        problem=PredictionProblem.regression,
    ),
    PredictionMethod(
        model=catboost.CatBoostClassifier,
        name="CatBoost",
        problem=PredictionProblem.classification,
    ),
    PredictionMethod(
        model=catboost.CatBoostRegressor,
        name="CatBoost",
        problem=PredictionProblem.regression,
    ),
    PredictionMethod(
        model=lightgbm.LGBMClassifier,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
        problem=PredictionProblem.classification,
    ),
    PredictionMethod(
        model=lightgbm.LGBMRegressor,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
        problem=PredictionProblem.regression,
    ),
    PredictionMethod(
        model=sklearn.ensemble.RandomForestClassifier,
        name="Random Forest",
        problem=PredictionProblem.classification,
    ),
    PredictionMethod(
        model=sklearn.ensemble.RandomForestRegressor,
        name="Random Forest",
        problem=PredictionProblem.regression,
    ),
]


class SelectionModel(Protocol):
    """Any sklearn selector."""

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        /,
    ) -> Any:
        ...

    def transform(
        self,
        X: numpy.ndarray,
        /,
    ) -> Any:
        ...


AnySelectionModel = TypeVar("AnySelectionModel", bound=SelectionModel)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SelectionMethod(Generic[AnySelectionModel]):
    """A method of solving feature selection problems.

    :param mask: Extract the feature selection mask from the model.
    :param model: Model constructor.
    :param name: Name of this method.
    """

    mask: Callable[[AnySelectionModel], numpy.ndarray]
    model: Callable[[PredictionModel], AnySelectionModel]
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name


SELECTION_METHODS = [
    SelectionMethod(
        mask=lambda model: model.get_support(),
        model=sklearn.feature_selection.SelectFromModel,
        name="Feature Importance",
    ),
    SelectionMethod(
        mask=lambda model: model.get_support(),
        model=sklearn.feature_selection.RFE,  # pyright: ignore[reportArgumentType]
        name="Recursive Feature Elimination",
    ),
    SelectionMethod(
        mask=lambda model: model.support_,
        model=boruta.BorutaPy,
        name="Boruta",
    ),
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainedModel:
    """A prediction model trained on select features in a ``dataset``.

    :param dataset: Dataset from which features are extracted.
    :param features: Features that the model was trained and tested on.
    :param prediction_method: Method of prediction.
    :param prediction_model: Model used to predict the target variable given the input variables.
    :param selection_method: Method of selection.
    :param selection_model: Model used to select relevant features from the ``prediction_model``.
    :param X_test: Input variables that the ``prediction_model`` was tested on.
    :param X_train: Input variables that the ``prediction_model`` was trained on.
    :param y_pred: Target variable that the ``prediction_model`` predicted during the test.
    :param y_test: Target variable that the ``prediction_model`` was tested on.
    :param y_train: Input variable that the ``prediction_model`` was trained on.
    """

    dataset: Dataset
    features: polars.DataFrame
    prediction_model: PredictionModel
    prediction_method: PredictionMethod
    selection_model: SelectionModel
    selection_method: SelectionMethod
    X_test: numpy.ndarray
    X_train: numpy.ndarray
    y_pred: numpy.ndarray
    y_test: numpy.ndarray
    y_train: numpy.ndarray

    def predict(
        self,
        known: IntoDataFrame,
    ) -> numpy.ndarray:
        """Predict the target variable given the ``known`` information.

        :param known: Data that is already known.
        :return: Target variable.
        """
        features = self.dataset.features(known)
        return self.prediction_model.predict(features.to_numpy())
