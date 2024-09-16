from __future__ import annotations

import abc
from typing import Iterable, TYPE_CHECKING

import polars

from autofeat.transform.identity import Identity
from autofeat.dataset.derived_dataset import DerivedDataset


if TYPE_CHECKING:
    from autofeat.transform.filter import Filter
    from autofeat.transform.base import Transform
    from autofeat.table import Table


class Dataset(abc.ABC):
    """A collection of tables."""

    @abc.abstractmethod
    def tables(
        self,
    ) -> Iterable[Table]:
        """List all tables in this dataset.

        :return: All tables.
        """

    def apply(
        self,
        transform: Transform,
        /,
    ) -> Dataset:
        """Apply the ``transform`` to each table in this dataset.

        :param transform: Transform to apply.
        :return: Derived dataset.
        """
        return DerivedDataset(
            dataset=self,
            transform=transform,
        )

    def features(
        self,
        *,
        filters: list[Filter] | None = None,
    ) -> polars.LazyFrame:
        """Extract features for each of the ``filters``.

        Features are the boolean or numeric columns from tables containing a single row. A simple
        way to guarantee that every table has a single row is to aggregate them by the same
        columns that the tables are filtered by.

        :param filters: Filters to apply before transforming the data.
        :return: Extracted features.
        """
        return polars.concat(
            [
                polars.concat(
                    [
                        table.data
                        .filter(polars.len() == 1)
                        .select(
                            (polars.selectors.boolean() | polars.selectors.numeric())
                            .name.suffix(f" from {table.name}"),
                        )
                        for table in filter.apply(self.tables())
                    ],
                    how="horizontal",
                )
                for filter in (filters or [Identity()])
            ],
            how="diagonal",
        )
