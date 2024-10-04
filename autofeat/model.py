from __future__ import annotations

import dataclasses
import enum
import functools
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar

import boruta
import catboost
import lightgbm
import numpy
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
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

    @functools.cached_property
    def baseline_method(
        self,
    ) -> PredictionMethod:
        """Get the baseline method for this kind of problem.

        :return: Baseline method.
        """
        match self:
            case PredictionProblem.classification:
                return PREDICTION_METHODS["random_guess"]
            case PredictionProblem.regression:
                return PREDICTION_METHODS["mean"]
            case _:
                raise NotImplementedError(f"{self} is not supported")


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


PREDICTION_METHODS: Final[dict[str, PredictionMethod]] = {
    "xgboost_classifier": PredictionMethod(
        model=xgboost.XGBClassifier,
        name="XGBoost",
        problem=PredictionProblem.classification,
    ),
    "xgboost_regressor": PredictionMethod(
        model=xgboost.XGBRegressor,
        name="XGBoost",
        problem=PredictionProblem.regression,
    ),
    "catboost_classifier": PredictionMethod(
        model=catboost.CatBoostClassifier,
        name="CatBoost",
        problem=PredictionProblem.classification,
    ),
    "catboost_regressor": PredictionMethod(
        model=catboost.CatBoostRegressor,
        name="CatBoost",
        problem=PredictionProblem.regression,
    ),
    "lightgbm_classifier": PredictionMethod(
        model=lightgbm.LGBMClassifier,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
        problem=PredictionProblem.classification,
    ),
    "lightgbm_regressor": PredictionMethod(
        model=lightgbm.LGBMRegressor,  # pyright: ignore[reportArgumentType]
        name="LightGBM",
        problem=PredictionProblem.regression,
    ),
    "linear_regression": PredictionMethod(
        model=sklearn.linear_model.LinearRegression,
        name="Linear Regression",
        problem=PredictionProblem.regression,
    ),
    "mean": PredictionMethod(
        model=lambda: sklearn.dummy.DummyRegressor(strategy="mean"),  # pyright: ignore[reportArgumentType]
        name="Mean",
        problem=PredictionProblem.regression,
    ),
    "most_frequent": PredictionMethod(
        model=lambda: sklearn.dummy.DummyClassifier(strategy="most_frequent"),
        name="Most Frequent",
        problem=PredictionProblem.classification,
    ),
    "random_forest_classifier": PredictionMethod(
        model=sklearn.ensemble.RandomForestClassifier,
        name="Random Forest",
        problem=PredictionProblem.classification,
    ),
    "random_forest_regressor": PredictionMethod(
        model=sklearn.ensemble.RandomForestRegressor,
        name="Random Forest",
        problem=PredictionProblem.regression,
    ),
    "random_guess": PredictionMethod(
        model=lambda: sklearn.dummy.DummyClassifier(strategy="uniform"),
        name="Random Guess",
        problem=PredictionProblem.classification,
    ),
}


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


SELECTION_METHODS: Final[dict[str, SelectionMethod]] = {
    "feature_importance": SelectionMethod(
        mask=lambda model: model.get_support(),
        model=lambda model: sklearn.feature_selection.SelectFromModel(model, max_features=50),
        name="Feature Importance",
    ),
    "recursive_feature_elimination": SelectionMethod(
        mask=lambda model: model.get_support(),
        model=sklearn.feature_selection.RFE,  # pyright: ignore[reportArgumentType]
        name="Recursive Feature Elimination",
    ),
    "boruta": SelectionMethod(
        mask=lambda model: model.support_,
        model=boruta.BorutaPy,
        name="Boruta",
    ),
}


class Transformer(Protocol):
    """Any invertible sklearn transformer."""

    def fit_transform(
        self,
        y: numpy.ndarray,
        /,
    ) -> Any:
        ...

    def inverse_transform(
        self,
        y: numpy.ndarray,
        /,
    ) -> numpy.ndarray:
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainedModel:
    """A prediction model trained on select features in a ``dataset``.

    :param baseline_model: Model used to benchmark the performance of this model.
    :param dataset: Dataset from which features are extracted.
    :param prediction_method: Method of prediction.
    :param prediction_model: Model used to predict the target variable given the input variables.
    :param selection_method: Method of selection.
    :param selection_model: Model used to select relevant features from the ``prediction_model``.
    :param transformer: Transformation applied by the model to the target variable.
    :param X: Input variables.
    :param X_test: Input variables used to test this model.
    :param X_train: Input variables used to train this model.
    :param y: Target variable.
    :param y_baseline: Target variable predicted by the baseline model on the test input variables.
    :param y_predicted: Target variable predicted by this model on the test input variables.
    :param y_test: Target variable used to test this model.
    :param y_train: Input variable used to train this model.
    """

    baseline_model: PredictionModel
    dataset: Dataset
    prediction_model: PredictionModel
    prediction_method: PredictionMethod
    selection_model: SelectionModel
    selection_method: SelectionMethod
    transformer: Transformer

    X: polars.DataFrame
    X_test: numpy.ndarray
    X_train: numpy.ndarray
    y: polars.Series
    y_baseline: numpy.ndarray
    y_predicted: numpy.ndarray
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
        X = self.dataset.features(known)
        y = self.prediction_model.predict(X.to_numpy())
        return self.transformer.inverse_transform(y)
