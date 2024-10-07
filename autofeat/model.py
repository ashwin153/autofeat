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
import sklearn.pipeline
import sklearn.preprocessing
import xgboost

from autofeat.convert import into_data_frame
from autofeat.transform import Aggregate, Collect, Combine, Drop, Extract, Identity, Keep

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy
    import polars

    from autofeat.convert import IntoDataFrame
    from autofeat.dataset import Dataset
    from autofeat.table import Column, Table


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
                return PREDICTION_METHODS["most_frequent"]
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
    model: Callable[[PredictionModel, int], AnySelectionModel]
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name


SELECTION_METHODS: Final[dict[str, SelectionMethod]] = {
    "feature_importance": SelectionMethod(
        mask=lambda model: model.get_support(),
        model=lambda model, n: sklearn.feature_selection.SelectFromModel(model, max_features=n),
        name="Feature Importance",
    ),
    "recursive_feature_elimination": SelectionMethod(
        mask=lambda model: model.get_support(),
        model=lambda model, n: sklearn.feature_selection.RFE(model, n_features_to_select=n),  # pyright: ignore[reportArgumentType]
        name="Recursive Feature Elimination",
    ),
    "boruta": SelectionMethod(
        mask=lambda model: model.support_,
        model=lambda model, n: boruta.BorutaPy(model),  # TODO: support n
        name="Boruta",
    ),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Model:  # type: ignore[no-any-unimported]
    """A prediction model trained on select features in a ``dataset``.

    :param baseline_model: Model used to benchmark the performance of this model.
    :param dataset: Dataset from which features are extracted.
    :param prediction_method: Method of prediction.
    :param prediction_model: Model used to predict the target variable given the input variables.
    :param selection_method: Method of selection.
    :param selection_model: Model used to select relevant features from the ``prediction_model``.
    :param X_test: Input variables used to test this model.
    :param X_train: Input variables used to train this model.
    :param X_transformer: Transformation applied by the model to the input variables.
    :param X: Input variables.
    :param y_baseline: Target variable predicted by the baseline model on the test input variables.
    :param y_predicted: Target variable predicted by this model on the test input variables.
    :param y_test: Target variable used to test this model.
    :param y_train: Input variable used to train this model.
    :param y_transformer: Transformation applied by the model to the target variable.
    :param y: Target variable.
    """

    baseline_model: PredictionModel
    dataset: Dataset
    prediction_model: PredictionModel
    prediction_method: PredictionMethod
    selection_model: SelectionModel
    selection_method: SelectionMethod
    X_test: numpy.ndarray
    X_train: numpy.ndarray
    X_transformer: sklearn.pipeline.Pipeline  # type: ignore[no-any-unimported]
    X: polars.DataFrame
    y_baseline: numpy.ndarray
    y_predicted: numpy.ndarray
    y_test: numpy.ndarray
    y_train: numpy.ndarray
    y_transformer: sklearn.base.TransformerMixin  # type: ignore[no-any-unimported]
    y: polars.Series

    def predict(
        self,
        known: IntoDataFrame,
    ) -> numpy.ndarray:
        """Predict the target variable given the ``known`` information.

        :param known: Data that is already known.
        :return: Target variable.
        """
        features = self.dataset.features(known)
        X = self.X_transformer.transform(features.to_numpy())
        y = self.prediction_model.predict(X)
        return self.y_transformer.inverse_transform(y)  # type: ignore[no-any-return]

    @staticmethod
    def train(
        dataset: Dataset,
        *,
        known_columns: tuple[Column, ...],
        prediction_method: PredictionMethod,
        training_data: Table,
        target_column: Column,
    ) -> Model:
        """

        :param dataset:
        :param known_columns:
        :param prediction_method:
        :param training_data:
        :param target_column:
        :return:
        """
        # extract the known and target variables from the training data
        known = (
            training_data.data
            .select([column.name for column in known_columns])
            .collect()
        )

        target = (
            training_data.data
            .select(target_column.name)
            .collect()
            .to_series()
        )

        # drop all columns related to the target column from the dataset
        dataset = dataset.apply(
            Drop(
                columns=[
                    (column, table)
                    for table in dataset.tables
                    for column in table.columns
                    if column.is_related(target_column)
                ],
            ),
        )

        # repeatedly transform the dataset and train a model on the n most important features
        iterations = [
            (
                [Aggregate(is_pivotable=known_columns)],
                SELECTION_METHODS["feature_importance"],
                160,
            ),
            (
                [],
                SELECTION_METHODS["recursive_feature_elimination"],
                80,
            ),
            (
                [Combine()],
                SELECTION_METHODS["recursive_feature_elimination"],
                40,
            ),
        ]


        i = 0
        while True:
            transform, selection_method, n_features = iterations[i]

            dataset = dataset.apply(Identity().then(Identity(), *transform).then(Collect()))

            model = Model._train_once(
                dataset=dataset,
                known=known,
                target=target,
                n_features=n_features,
                prediction_method=prediction_method,
                selection_method=selection_method,
            )

            dataset = model.dataset

            if i == len(iterations) - 1:
                return model
            else:
                i += 1

    @staticmethod
    def _train_once(
        *,
        dataset: Dataset,
        known: polars.DataFrame,
        target: polars.Series,
        n_features: int = 25,
        prediction_method: PredictionMethod,
        selection_method: SelectionMethod,
    ) -> Model:
        # extract features from the dataset
        features = dataset.apply(Extract(known=known))

        # pre-process the input and target variables and split them into training and test data
        X = into_data_frame(features)
        y = target

        X_transformer = sklearn.pipeline.Pipeline([
            # TODO: create a custom scaler that only applies to numeric columns
            # ("scale", sklearn.preprocessing.RobustScaler()),
            ("identity", sklearn.preprocessing.FunctionTransformer()),
        ])

        y_transformer = (
            sklearn.preprocessing.LabelEncoder()
            if prediction_method.problem == PredictionProblem.classification
            else sklearn.preprocessing.FunctionTransformer()
        )

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X_transformer.fit_transform(X.to_numpy()),
            y_transformer.fit_transform(y.to_numpy()),
        )

        # create a prediction model
        prediction_model = prediction_method.model()

        # train a model that selects the n most important features to the prediction model
        selection_model = selection_method.model(prediction_model, n_features)
        selection_model.fit(X_train, y_train)

        # drop features that were not selected from the training data
        X_train = selection_model.transform(X_train)
        X_test = selection_model.transform(X_test)
        X_transformer.fit_transform(X_train)
        X = X.select(c for c, x in zip(X.columns, selection_method.mask(selection_model)) if x)

        # keep only the columns that selected features are extracted from
        dataset = dataset.apply(
            Keep(
                columns=[
                    ancestor
                    for table in features.tables
                    for column in table.columns
                    if column.name in X.columns
                    for ancestor in column.derived_from
                ],
            ),
        )

        # train the prediction model on the selected features
        prediction_model.fit(X_train, y_train)

        # evaluate the prediction model on the test data
        y_predicted = prediction_model.predict(X_test)

        # train the baseline model on the selected features
        baseline_model = prediction_method.problem.baseline_method.model()
        baseline_model.fit(X_train, y_train)

        # evaluate the baseline model on the test data
        y_baseline = baseline_model.predict(X_test)

        # collect all the intermediate outputs
        return Model(
            baseline_model=baseline_model,
            dataset=dataset,
            prediction_method=prediction_method,
            prediction_model=prediction_model,
            selection_method=selection_method,
            selection_model=selection_model,
            X_test=X_transformer.inverse_transform(X_test),
            X_train=X_transformer.inverse_transform(X_train),
            X_transformer=X_transformer,
            X=X,
            y_baseline=y_transformer.inverse_transform(y_baseline),
            y_predicted=y_transformer.inverse_transform(y_predicted),
            y_test=y_transformer.inverse_transform(y_test),
            y_train=y_transformer.inverse_transform(y_train),
            y_transformer=y_transformer,
            y=y,
        )
