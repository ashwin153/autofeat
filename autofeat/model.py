from __future__ import annotations

import dataclasses
import enum
import functools
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar

import boruta
import catboost
import lightgbm
import loguru
import numpy
import pandas
import shap
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
from autofeat.transform import Aggregate, Combine, Drop, Extract, Identity, Keep, Transform

if TYPE_CHECKING:
    from collections.abc import Callable

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
                return PREDICTION_METHODS["most_frequent_category"]
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
        model=lambda: xgboost.XGBClassifier(device="cuda"),
        name="XGBoost",
        problem=PredictionProblem.classification,
    ),
    "xgboost_regressor": PredictionMethod(
        model=lambda: xgboost.XGBRegressor(device="cuda"),
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
    "most_frequent_category": PredictionMethod(
        model=lambda: sklearn.dummy.DummyClassifier(strategy="most_frequent"),
        name="Most Frequent Category",
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
    "random_category": PredictionMethod(
        model=lambda: sklearn.dummy.DummyClassifier(strategy="uniform"),
        name="Random Category",
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


class AutofeatSelector(
    sklearn.base.BaseEstimator,  # type: ignore[no-any-unimported]
    sklearn.feature_selection.SelectorMixin,  # type: ignore[no-any-unimported]
):
    """Select the features with the highest SHAP values that are not correlated with other features.

    :param max_correlation: Maximum correlation that a selected feature can have with any feature.
    :param model: Model to select features from.
    :param num_features: Number of features to select.
    :param num_samples: Number of samples to use for the correlation and SHAP calculations.
    """

    def __init__(
        self,
        *,
        max_correlation: float = 0.4,
        model: PredictionModel,
        num_features: int,
        num_samples: int = 2500,
    ) -> None:
        self._max_correlation = max_correlation
        self._model = model
        self._num_features = num_features
        self._num_samples = num_samples
        self._support_mask: numpy.ndarray | None = None

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        /,
    ) -> Any:
        # fit the model
        self._model.fit(X, y)

        # find columns that are highly correlated
        correlated = (
            numpy.max(
                numpy.triu(
                    numpy.abs(
                        # TODO: use numpy.ma.corrcoeff and numpy.ma.masked_invalid
                        pandas.DataFrame(X[:self._num_samples, :]).corr().to_numpy(),
                    ),
                    k=1,
                ),
                axis=1,
            ) > self._max_correlation
        )

        # find the shap values associated with the model
        explanation = shap.Explainer(self._model)(X[:self._num_samples, :])

        # determine the shap importance of each column
        importance = (
            numpy
            .abs(explanation.values)
            .mean(tuple(i for i in range(len(explanation.shape)) if i != 1))
        )

        selection = (
            numpy
            .where(correlated, 0, importance)
            .argpartition(-self._num_features)[-self._num_features:]
        )

        # construct a bitmask from the selected columns
        self._support_mask = numpy.array([i in selection for i in range(X.shape[1])])

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
    "autofeat": SelectionMethod(
        mask=lambda model: model.get_support(),
        model=lambda model, n: AutofeatSelector(model=model, num_features=n),
        name="autofeat",
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

    @functools.cached_property
    def explanation(  # type: ignore[no-any-unimported]
        self,
    ) -> shap.Explanation:
        """Get the SHAP explanation of this model.

        :return: SHAP explanation.
        """
        explainer = shap.Explainer(
            self.prediction_model,
            feature_names=self.X.columns,
        )

        return explainer(self.X_test)

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
        """Train a model that predicts the ``target_column`` given the ``known_columns``.

        :param dataset: Dataset to extract features from.
        :param known_columns: Columns that are known at the time of prediction.
        :param prediction_method: Method of predicting the target variable.
        :param training_data: Table containing the ``target_column`` and ``known_columns``.
        :param target_column: Column containing the target variable.
        :return: Trained model.
        """
        # extract the known and target variables from the training data
        loguru.logger.info("loading training data")

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

        # repeatedly transform the dataset and train a prediction model on the top n features
        iterations: list[tuple[list[Transform], SelectionMethod, int]] = [
            (
                [Aggregate(is_pivotable=known_columns)],
                SELECTION_METHODS["feature_importance"],
                150,
            ),
            (
                [],
                SELECTION_METHODS["autofeat"],
                100,
            ),
            (
                [Combine()],
                SELECTION_METHODS["autofeat"],
                50,
            ),
        ]

        i = 0
        while True:
            loguru.logger.info(f"training model ({i+1}/{len(iterations)})")

            transforms, selection_method, num_features = iterations[i]

            dataset = dataset.apply(Identity().then(Identity(), *transforms))

            model = Model._train_once(
                dataset=dataset,
                known=known,
                num_features=num_features,
                prediction_method=prediction_method,
                selection_method=selection_method,
                target=target,
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
        num_features: int,
        prediction_method: PredictionMethod,
        selection_method: SelectionMethod,
        target: polars.Series,
    ) -> Model:
        # extract features from the dataset
        loguru.logger.info("extracting features")

        features = dataset.apply(Extract(known=known))

        X = into_data_frame(features)
        y = target

        # pre-process the input and target variables and split them into training and test data
        loguru.logger.info("splitting training data")

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
            train_size=0.8,
            shuffle=False,
        )

        # create prediction and selection models
        prediction_model = prediction_method.model()
        selection_model = selection_method.model(prediction_model, num_features)

        if num_features < X.shape[1]:
            # train the selection model
            loguru.logger.info("fitting selection model")

            selection_model.fit(X_train, y_train)

            # apply feature selection to the training and test data
            loguru.logger.info("applying feature selection")

            X_train = selection_model.transform(X_train)
            X_test = selection_model.transform(X_test)
            X_transformer.fit(X_train)
            X = X.select(c for c, x in zip(X.columns, selection_method.mask(selection_model)) if x)

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
        loguru.logger.info("fitting prediction model")

        prediction_model.fit(X_train, y_train)

        # evaluate the prediction model on the test data
        loguru.logger.info("evaluating prediction model")

        y_predicted = prediction_model.predict(X_test)

        # train the baseline model on the selected features
        loguru.logger.info("fitting baseline model")

        baseline_model = prediction_method.problem.baseline_method.model()
        baseline_model.fit(X_train, y_train)

        # evaluate the baseline model on the test data
        loguru.logger.info("evaluating baseline model")

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
