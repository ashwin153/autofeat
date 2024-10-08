import dataclasses
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert.into_exprs import into_exprs
from autofeat.convert.into_named_exprs import into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Impute(Transform):
    """Fill missing data in different ways."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if imputations := list(self._imputations(table)):
                extra_columns = [
                    column
                    for column in table.columns
                    if Attribute.primary_key in column.attributes
                ]

                columns = [
                    *extra_columns,
                    *[column for column, _ in imputations],
                ]

                yield Table(
                    data=table.data.select(
                        *into_exprs(extra_columns),
                        **into_named_exprs(imputations),
                    ),
                    name=f"impute({table.name})",
                    columns=columns,
                )

    def _imputations(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, polars.Expr]]:
        numeric_columns = [
            column
            for column in table.columns
            if Attribute.numeric in column.attributes
            if Attribute.not_null not in column.attributes
            if Attribute.categorical not in column.attributes
        ]

        for x in numeric_columns:
            imputations = [
                (f"impute_mean({x})", x.expr.fill_null(strategy="mean")),
            ]

            for name, expr in imputations:
                column = Column(
                    name=name,
                    attributes=x.attributes | {Attribute.not_null},
                    derived_from=[(x, table)],
                )

                yield column, expr
