from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars

from autofeat.attribute import Attribute

if TYPE_CHECKING:
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
        where: polars.DataFrame,
    ) -> polars.DataFrame:
        """Extract features from all tables in this dataset that are relevant to ``where``.

        .. note::

            Feature extraction is a computationally expensive operation.

        :param where: Where clause.
        :return: Extracted features.
        """
        features = [
            (
                where
                .lazy()
                .join(table.data, on=list(primary_key), how="left")
                .select(polars.selectors.by_name(set(table.schema) - primary_key))
                .select(polars.selectors.boolean() | polars.selectors.numeric())
                .select(polars.all().name.suffix(f" from {table.name}"))
            )
            for table in self.tables
            if (primary_key := set(table.schema.select(include={Attribute.primary_key})))
            if primary_key.issubset(where.columns)
        ]

        return polars.concat(
            polars.collect_all(features),
            how="horizontal",
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
