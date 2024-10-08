import dataclasses
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert.into_exprs import into_exprs
from autofeat.convert.into_named_exprs import into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform

# Number of rows to test casts on.
SAMPLE_SIZE = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cast(Transform):
    """Cast columns to more appropriate types.

    In particular, string columns containing date, time, or datetime values are converted to
    temporal columns. Casting enables further transformation (e.g., time columns enable rolling
    aggregation) and improves performance (e.g., categorical columns are easier to join).
    """

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        tables = list(tables)

        samples = polars.collect_all([
            (
                table.data
                .select(**into_named_exprs(self._castable_columns(table)))
                .head(SAMPLE_SIZE)
            )
            for table in tables
        ])

        for table, sample in zip(tables, samples):
            casts = [
                self._cast(table, sample, column)
                for column in self._castable_columns(table)
            ]

            extra_columns = [
                column
                for column in table.columns
                if all(column.name != casted_column.name for casted_column, _ in casts)
            ]

            columns = [
                *extra_columns,
                *[column for column, _ in casts],
            ]

            yield Table(
                data=table.data.select(*into_exprs(extra_columns), **into_named_exprs(casts)),
                name=table.name,
                columns=columns,
            )

    def _castable_columns(
        self,
        table: Table,
    ) -> list[Column]:
        return [
            column
            for column in table.columns
            if Attribute.textual in column.attributes
            if Attribute.not_null in column.attributes
        ]

    def _cast(
        self,
        table: Table,
        sample: polars.DataFrame,
        column: Column,
    ) -> tuple[Column, polars.Expr]:
        casts = [
            (column.expr.str.to_date("%Y-%m-%d"), {Attribute.temporal}),
            (column.expr.str.to_datetime(), {Attribute.temporal}),
            (column.expr.str.to_time(), {Attribute.temporal}),
        ]

        for expr, attributes in casts:
            try:
                sample.select(expr)
            except polars.exceptions.PolarsError:
                pass
            else:
                column = Column(
                    name=column.name,
                    attributes=attributes,
                    derived_from=[(column, table)],
                )

                return column, expr

        return column, column.expr
