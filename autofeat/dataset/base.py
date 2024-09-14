from __future__ import annotations

import abc
from typing import Iterable, TYPE_CHECKING

import polars

from autofeat.transform.identity import Identity


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

    def features(
        self,
        *,
        filters: list[Filter] | None = None,
        transform: Transform | None = None,
    ) -> polars.LazyFrame:
        """Extract features under the ``transform`` given each of the ``filters``.

        Features are the boolean or numeric columns from tables that reduce to a single row under
        the ``transform``. A simple way to guarantee this is to always filter and aggregate data by
        the same columns.

        :param filters: Filters to apply before transforming the data.
        :param transform: Transform to apply to the data.
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
                        for table in filter.then(transform or Identity()).apply(self.tables())
                    ],
                    how="horizontal",
                )
                for filter in (filters or [Identity()])
            ],
            how="diagonal",
        )
