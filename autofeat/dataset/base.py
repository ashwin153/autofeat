from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars

from autofeat.transform.identity import Identity

if TYPE_CHECKING:
    from autofeat.table import Table
    from autofeat.transform.base import Transform
    from autofeat.transform.filter import Filter


class Dataset(abc.ABC):
    """A collection of tables."""

    @abc.abstractmethod
    def tables(
        self,
    ) -> Iterable[Table]:
        """List all tables in this dataset.

        :return: All tables.
        """

    def derive(
        self,
        transform: Transform,
        /,
    ) -> Dataset:
        """Apply the ``transform`` to each table in this dataset.

        :param transform: Transform to apply.
        :return: Derived dataset.
        """
        from autofeat.dataset.derived_dataset import DerivedDataset

        return DerivedDataset(dataset=self, transform=transform)

    def features(
        self,
        *filters: Filter | Iterable[Filter],
    ) -> polars.LazyFrame:
        """Extract features for each of the ``filters``.

        Features are the boolean or numeric columns from tables containing a single row. A simple
        way to guarantee every table has a single row is to aggregate tables by the same columns
        that they are filtered by.

        :param filters: Filters to apply before transforming the data.
        :return: Extracted features.
        """
        transforms: list[Transform] = [
            y
            for x in (filters or (Identity(),))
            for y in (x if isinstance(x, Iterable) else (x,))
        ]

        feature_vectors = []

        for transform in transforms:
            feature_values = []

            for table in transform.apply(self.tables()):
                feature_selector = (
                    (polars.selectors.boolean() | polars.selectors.numeric())
                    .name.suffix(f" from {table.name}")
                )

                feature_values.append(
                    table.data
                    .filter(polars.len() == 1)
                    .select(feature_selector),
                )

            feature_vectors.append(polars.concat(feature_values, how="horizontal"))

        return polars.concat(feature_vectors, how="diagonal")

    def merge(
        self,
        *datasets: Dataset,
    ) -> Dataset:
        """Add the ``datasets`` to this dataset.

        :param datasets: Datasets to merge.
        :return: Merged dataset.
        """
        from autofeat.dataset.merged_dataset import MergedDataset

        if datasets:
            return MergedDataset(datasets=[self, *datasets])
        else:
            return self
