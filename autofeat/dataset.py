from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars

from autofeat.transform.extract import Extract

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

    def extract(
        self,
        *,
        given: IntoDataFrame,
    ) -> polars.DataFrame:
        """Extract features from all tables in this dataset that are relevant to the ``given`` data.

        .. note::

            Feature extraction is a computationally expensive operation.

        :param where: Where clause.
        :return: Extracted features.
        """
        features = [
            table.data.select(polars.all().name.suffix(f" from {table.name}"))
            for table in Extract(given=given).apply(self.tables)
        ]

        return polars.concat(polars.collect_all(features), how="horizontal")

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
