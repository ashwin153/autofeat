import dataclasses
from collections.abc import Iterable
from typing import ClassVar

import polars

from autofeat.attribute import Attribute
from autofeat.convert import (
    IntoDataFrame,
    IntoSeries,
    into_data_frame,
    into_named_exprs,
    into_series,
)
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Extract(Transform):
    """Extract features that are relevant to the ``known`` data.

    :param as_of: Time as of which to extract features.
    :param known: Data that is already known.
    """

    # Reserved characters used to separate column and table names.
    SEPARATOR: ClassVar = " :: "

    as_of: IntoSeries | None = None
    known: IntoDataFrame

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        as_of = None if self.as_of is None else into_series(self.as_of)
        known = into_data_frame(self.known)

        for table in tables:
            primary_key = {
                column.name
                for column in table.columns
                if Attribute.primary_key in column.attributes
            }

            features = sorted(
                self._features(table),
                key=lambda feature: len(feature[0].name),
            )

            if (
                primary_key
                and primary_key.issubset(known.columns)
                and features
            ):
                columns = [
                    column
                    for column, _ in features
                ]

                if time_column := self._time_column(table):
                    data = (
                        known
                        .with_columns(**{time_column.name: as_of})
                        .lazy()
                        .join_asof(table.data, on=time_column.name, by=list(primary_key))
                        .select(**into_named_exprs(features))
                    )
                else:
                    data = (
                        known
                        .lazy()
                        .join(table.data, on=list(primary_key), how="left")
                        .select(**into_named_exprs(features))
                    )

                yield Table(
                    columns=columns,
                    data=data,
                    name=f"features({table.name})",
                )

    def _features(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, polars.Expr]]:
        primary_key = [
            (column, table)
            for column in table.columns
            if Attribute.primary_key in column.attributes
        ]

        for x in table.columns:
            if  (
                {Attribute.boolean, Attribute.numeric} & x.attributes
                and Attribute.primary_key not in x.attributes
            ):
                column = Column(
                    name=f"{x.name}{Extract.SEPARATOR}{table.name}",
                    attributes=x.attributes,
                    derived_from=[(x, table), *primary_key],
                )

                expr = (
                    x.expr.cast(polars.Boolean())
                    if Attribute.boolean in x.attributes
                    else x.expr.shrink_dtype()
                )

                yield column, expr

    def _time_column(
        self,
        table: Table,
    ) -> Column | None:
        for column in table.columns:
            if Attribute.temporal in column.attributes:
                return column

        return None
