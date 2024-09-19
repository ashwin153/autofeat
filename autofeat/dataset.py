from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars

from autofeat.convert.into_filters import IntoFilters, into_filters
from autofeat.transform.filter import Filter
from autofeat.transform.identity import Identity

if TYPE_CHECKING:
    from collections.abc import Iterable

    from autofeat.table import Table

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        *,
        filters: IntoFilters | Iterable[IntoFilters] | None = None,
        transform: Transform | None = None,
    ) -> polars.LazyFrame:
        """Extract features under the ``transform`` for each of the ``filters``.

        Features are the boolean or numeric columns in tables that reduce to a single row under the
        ``transform``. A simple way to guarantee that every table has a single row is to aggregate
        tables by the same columns that are used in ``filters``.

        :param filters: Filters to apply before transforming the data.
        :param transform: Transform to apply to tables.
        :return: Extracted features.
        """
        feature_vectors = []

        filters = (
            (Filter(),)
            if filters is None
            else into_filters(filters)
        )

        for filter in filters:
            feature_values = []

            tables = (
                filter
                .then(transform or Identity())
                .apply(self.tables)
            )

            for table in tables:
                feature_selector = (
                    (polars.selectors.boolean() | polars.selectors.numeric())
                    .name.suffix(f" from {table.name}")
                )

                feature_values.append(
                    table.data
                    .filter(polars.len() == 1)
                    .select(feature_selector),
                )

            if feature_values:
                feature_vectors.append(
                    polars.concat(
                        feature_values,
                        how="horizontal",
                    ),
                )

        if feature_vectors:
            return polars.concat(
                feature_vectors,
                how="diagonal",
            )
        else:
            return polars.LazyFrame()

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
