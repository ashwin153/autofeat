from __future__ import annotations

import collections
import dataclasses
from typing import TYPE_CHECKING

import polars
import sklearn.model_selection

from autofeat.attribute import Attribute
from autofeat.convert import IntoDataFrame, IntoSeries, into_data_frame, into_series
from autofeat.model import (
    PREDICTION_METHODS,
    SELECTION_METHODS,
    PredictionMethod,
    SelectionMethod,
    TrainedModel,
)
from autofeat.transform.keep import Keep

if TYPE_CHECKING:
    from autofeat.convert import IntoDataFrame
    from autofeat.table import Table
    from autofeat.transform.base import Transform


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
                .select(polars.all().name.suffix(f"::{table.name}"))
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
        prediction_method: PredictionMethod = PREDICTION_METHODS[0],
        selection_method: SelectionMethod = SELECTION_METHODS[0],
    ) -> TrainedModel:
        """Train a model for predicting the ``target`` given the ``known`` information.

        :param known: Data that is already known.
        :param target: Target variable.
        :param prediction_method: Method of prediction.
        :param selection_method: Method of feature selection.
        :return: Trained model.
        """
        # load features from this dataset
        features = self.features(known)

        # split features and target into training and test data
        X = features.to_numpy()
        y = into_series(target).to_numpy()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

        # create a model that predicts the target from the features
        prediction_model = prediction_method.model()

        # train a model that selects the most important features to the prediction model
        selection_model = selection_method.model(prediction_model)
        selection_model.fit(X_train, y_train)

        # apply feature selection to the training and test data
        X_train = selection_model.transform(X_train)
        X_test = selection_model.transform(X_test)

        # apply feature selection to this dataset
        selection = collections.defaultdict(set)
        for i, selected in enumerate(selection_method.mask(selection_model)):
            if selected:
                column, table = features.columns[i].split("::", 1)
                selection[table].add(column)

        dataset = self.apply(Keep(columns=selection))

        # train the prediction model on the selected features
        prediction_model.fit(X_train, y_train)

        # evaluate the prediction model on the test data
        y_pred = prediction_model.predict(X_test)

        # collect all the intermediate outputs
        return TrainedModel(
            dataset=dataset,
            features=features,
            prediction_method=prediction_method,
            prediction_model=prediction_model,
            selection_method=selection_method,
            selection_model=selection_model,
            X_test=X_test,
            X_train=X_train,
            y_pred=y_pred,
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
