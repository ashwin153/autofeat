from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars

from autofeat.attribute import Attribute

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class Schema:
    """A description of the structure of a tabular dataset.

    :param columns: Columns in the table.
    """

    columns: dict[str, set[Attribute]]

    def __iter__(
        self,
    ) -> Iterable[tuple[str, set[Attribute]]]:
        return self.columns.items()

    def select(
        self,
        *,
        include: set[Attribute] | None = None,
        exclude: set[Attribute] | None = None,
    ) -> set[str]:
        """Select a subset of columns by attributes.

        :param include: Attributes that columns must have.
        :param exclude: Attributes that columns must not have.
        :return: Selected columns.
        """
        return {
            name
            for name, attributes in self.columns.items()
            if include is None or include.issubset(attributes)
            if exclude is None or exclude.isdisjoint(attributes)
        }

    @staticmethod
    def infer(
        data: polars.LazyFrame,
        /,
    ) -> Schema:
        """Infer the schema of the ``data``.

        .. note::

            Schema inference is a computationally expensive operation, because it requires
            collecting the schema and various summary statistics from the ``data``.

        :param data: Data to infer the schema of.
        :return: Inferred schema.
        """
        # profile the data
        metrics = {
            "len":
                data.select(polars.all().len()),
            "n_unique":
                data.select(polars.all().n_unique()),
            "null_count":
                data.select(polars.all().null_count()),
        }

        profile = {
            metric: df.row(0, named=True)
            for metric, df in zip(
                metrics.keys(),
                polars.collect_all(metrics.values()),
            )
        }

        # use the profile and the schema of the data to infer column attributes
        columns = {}

        for column, data_type in data.schema.items():
            attributes = set()

            if isinstance(data_type, polars.Boolean):
                attributes.add(Attribute.boolean)

            if profile["n_unique"][column] < 100:
                attributes.add(Attribute.categorical)

            if profile["null_count"][column] == 0:
                attributes.add(Attribute.not_null)

            if data_type.is_numeric():
                attributes.add(Attribute.numeric)

            if profile["n_unique"][column] < profile["len"][column] * 0.10:
                attributes.add(Attribute.pivotable)

            if profile["n_unique"][column] == profile["len"][column]:
                attributes.add(Attribute.primary_key)

            if isinstance(data_type, polars.String):
                attributes.add(Attribute.textual)

            columns[column] = attributes

        return Schema(columns)
