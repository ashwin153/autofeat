from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING

import polars
import sklearn.model_selection
import sklearn.preprocessing

from autofeat.attribute import Attribute
from autofeat.convert import IntoDataFrame, IntoSeries, into_data_frame, into_series
from autofeat.model import (
    PREDICTION_METHODS,
    SELECTION_METHODS,
    PredictionMethod,
    PredictionProblem,
    SelectionMethod,
    TrainedModel,
)
from autofeat.transform.keep import Keep

if TYPE_CHECKING:
    from autofeat.convert import IntoDataFrame
    from autofeat.table import Table
    from autofeat.transform.base import Transform


# Delimiter used to separate column and table names in feature names.
_SEPARATOR = " :: "


@dataclasses.dataclass(frozen=True)
class Dataset:
    """A collection of tables.

    :param tables: Tables in this dataset.
    """

    tables: list[Table]

    def apply(
        self,
        transform: Transform,
        /,
    ) -> Dataset:
        """Apply the ``transform`` to each table in this dataset.

        :param transform: Transform to apply.
        :return: Transformed dataset.
        """
        return Dataset(list(transform.apply(self.tables)))

    def features(
        self,
        known: IntoDataFrame,
    ) -> polars.DataFrame:
        """Extract features from all tables in this dataset that are relevant to the ``known`` data.

        .. note::

            Feature extraction is a computationally expensive operation.

        :param known: Data that is already known.
        :return: Extracted features.
        """
        known = into_data_frame(known)

        features = [
            (
                known
                .lazy()
                .join(table.data, on=list(primary_key), how="left")
                .select(polars.selectors.boolean() | polars.selectors.numeric())
                .select(polars.all().name.suffix(f"{_SEPARATOR}{table.name}"))
            )
            for table in self.tables
            if (primary_key := set(table.schema.select(include={Attribute.primary_key})))
            if primary_key.issubset(known.columns)
        ]

        return polars.concat(
            polars.collect_all(features),
            how="horizontal",
        )

    def train(
        self,
        known: IntoDataFrame,
        target: IntoSeries,
        *,
        prediction_method: PredictionMethod = PREDICTION_METHODS["xgboost_classifier"],
        selection_method: SelectionMethod = SELECTION_METHODS["feature_importance"],
    ) -> TrainedModel:
        """Train a model for predicting the ``target`` given the ``known`` information.

        :param known: Data that is already known.
        :param target: Target variable.
        :param prediction_method: Method of prediction.
        :param selection_method: Method of feature selection.
        :return: Trained model.
        """
        # split input and target variables into training and test data
        X = self.features(known)

        y = into_series(target).to_numpy()
        if prediction_method.problem == PredictionProblem.classification:
            label_encoder = sklearn.preprocessing.LabelEncoder()
            y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X.to_numpy(),
            y,
        )

        # train a model that selects the most important features to a prediction model
        prediction_model = prediction_method.model()

        selection_model = selection_method.model(prediction_model)
        selection_model.fit(X_train, y_train)

        # apply feature selection to the training and test data
        selection = [
            X.columns[i]
            for i, is_selected in enumerate(selection_method.mask(selection_model))
            if is_selected
        ]

        selection_by_table = collections.defaultdict(set)
        for selected in selection:
            column, table = selected.split(_SEPARATOR, 1)
            selection_by_table[table].add(column)

        X_train = selection_model.transform(X_train)
        X_test = selection_model.transform(X_test)
        X = X.select(selection)
        dataset = self.apply(Keep(columns=selection_by_table))

        # train the prediction model on the selected features
        prediction_model.fit(X_train, y_train)

        # collect all the intermediate outputs
        return TrainedModel(
            dataset=dataset,
            prediction_method=prediction_method,
            prediction_model=prediction_model,
            selection_method=selection_method,
            selection_model=selection_model,
            X=X,
            X_test=X_test,
            X_train=X_train,
            y=y,
            y_test=y_test,
            y_train=y_train,
        )

    def table(
        self,
        name: str,
    ) -> Table:
        """Get the table with the corresponding name.

        :param name: Name of the table.
        :return: Corresponding table.
        """
        for table in self.tables:
            if table.name == name:
                return table

        raise ValueError(f"table `{name}` does not exist")
