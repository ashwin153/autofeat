from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import attrs
import loguru
import polars
import shap
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from autofeat.convert import into_data_frame
from autofeat.problem import Problem
from autofeat.selector import Correlation, FeatureImportance, Selector, ShapelyImpact
from autofeat.transform import Aggregate, Drop, Extract, Filter, Identity, Keep, Transform

if TYPE_CHECKING:
    import numpy

    from autofeat.convert import IntoDataFrame
    from autofeat.dataset import Dataset
    from autofeat.predictor import PredictionMethod, Predictor
    from autofeat.table import Column, Table


@attrs.define(frozen=True, kw_only=True, slots=True)
class Prediction:
    """A prediction made by a model.

    :param model: Model that made the prediction.
    :param known: Data that was known at the time of prediction.
    :param X: Input variables.
    :param y: Target variable.
    """

    model: Model
    known: polars.DataFrame
    X: polars.DataFrame
    y: polars.Series

    @functools.cached_property
    def explanation(  # type: ignore[no-any-unimported]
        self,
    ) -> shap.Explanation:
        """Get the SHAP explanation of this prediction.

        :return: SHAP explanation.
        """
        explainer = shap.Explainer(
            self.model.predictor,
            feature_names=self.X.columns,
        )

        return explainer(self.X.to_numpy())


@attrs.define(frozen=True, kw_only=True, slots=True)
class Model:  # type: ignore[no-any-unimported]
    """A prediction model trained on select features in a ``dataset``.

    :param dataset: Dataset from which features are extracted.
    :param known: Data that was known at the time of feature extraction.
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

    dataset: Dataset
    known: polars.DataFrame
    predictor: Predictor
    problem: Problem
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
            self.predictor,
            feature_names=self.X.columns,
        )

        return explainer(self.X_test)

    def predict(
        self,
        known: IntoDataFrame,
    ) -> Prediction:
        """Predict the target variable given the ``known`` information.

        :param known: Data that is already known.
        :return: Target variable.
        """
        known = into_data_frame(known)
        assert set(known.columns) == set(self.known.columns)

        X = self.dataset.features(known)

        y = polars.Series(
            name=self.y.name,
            values=self.y_transformer.inverse_transform(  # pyright: ignore[reportArgumentType]
                self.predictor.predict(
                    self.X_transformer.transform(
                        X.to_numpy(),
                    ),
                ),
            ),
        )

        return Prediction(
            model=self,
            known=known,
            X=X,
            y=y,
        )

    @staticmethod
    def train(
        dataset: Dataset,
        *,
        known_columns: tuple[Column, ...],
        prediction_method: PredictionMethod,
        problem: Problem,
        target_column: Column,
        training_data: Table,
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

        # repeatedly transform the dataset and train a predictor on the top n features
        predictor = prediction_method.create(problem)

        iterations: list[tuple[list[Transform], list[Selector]]] = [
            (
                [
                    Aggregate(is_pivotable=known_columns, max_pivots=1),
                ],
                [
                    FeatureImportance(predictor=predictor, n=200),
                    Correlation(max=0.7),
                    ShapelyImpact(predictor=predictor, n=75),
                ],
            ),
            (
                [
                    Filter().then(Aggregate(is_pivotable=known_columns, max_pivots=1)),
                ],
                [
                    Correlation(max=0.5),
                    ShapelyImpact(predictor=predictor, n=50),
                ],
            ),
        ]

        i = 0
        while True:
            loguru.logger.info(f"training model ({i+1}/{len(iterations)})")

            transforms, selectors = iterations[i]

            dataset = dataset.apply(Identity().then(Identity(), *transforms))

            model = Model._train_once(
                dataset=dataset,
                known=known,
                predictor=predictor,
                problem=problem,
                selectors=selectors,
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
        predictor: Predictor,
        problem: Problem,
        selectors: list[Selector],
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
            if problem == Problem.classification
            else sklearn.preprocessing.FunctionTransformer()
        )

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X_transformer.fit_transform(X.to_numpy()),
            y_transformer.fit_transform(y.to_numpy()),
            train_size=0.8,
            shuffle=False,
        )

        # create prediction and selection models
        for i, selector in enumerate(selectors):
            # train the selection model
            loguru.logger.info(f"fitting selection model ({i+1}/{len(selectors)})")

            selector.fit(X_train, y_train)

            # apply feature selection to the training and test data
            loguru.logger.info("applying feature selection")

            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            X_transformer.fit(X_train)
            X = X.select(c for c, x in zip(X.columns, selector.get_support()) if x)

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

        predictor.fit(X_train, y_train)

        # evaluate the prediction model on the test data
        loguru.logger.info("evaluating prediction model")

        y_predicted = predictor.predict(X_test)

        # train the baseline model on the selected features
        loguru.logger.info("fitting baseline model")

        baseline_model = problem.baseline_method.model()
        baseline_model.fit(X_train, y_train)

        # evaluate the baseline model on the test data
        loguru.logger.info("evaluating baseline model")

        y_baseline = baseline_model.predict(X_test)

        # collect all the intermediate outputs
        return Model(
            baseline_model=baseline_model,
            dataset=dataset,
            known=known,
            problem=problem,
            predictor=predictor,
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
