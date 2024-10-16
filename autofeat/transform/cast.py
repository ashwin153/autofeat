from collections.abc import Iterable

import attrs
import polars

from autofeat.attribute import Attribute
from autofeat.convert.into_named_exprs import into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform

# Number of rows to test casts on.
SAMPLE_SIZE = 25


@attrs.define(frozen=True, kw_only=True, slots=True)
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
            table.data.head(SAMPLE_SIZE)
            for table in tables
        ])

        for table, sample in zip(tables, samples):
            casts = [
                self._cast(table, sample, column)
                for column in table.columns
            ]

            yield Table(
                data=table.data.select(**into_named_exprs(casts)),
                name=table.name,
                columns=[column for column, _ in casts],
            )

    def _cast(
        self,
        table: Table,
        sample: polars.DataFrame,
        column: Column,
    ) -> tuple[Column, polars.Expr]:
        if {Attribute.textual, Attribute.not_null} <= column.attributes:
            exprs = [
                column.expr.str.to_date("%Y-%m-%d"),
                column.expr.str.to_datetime(),
                column.expr.str.to_time(),
            ]

            for expr in exprs:
                try:
                    sample.select(expr)
                except polars.exceptions.PolarsError:
                    pass
                else:
                    result = Column(
                        name=column.name,
                        attributes={Attribute.temporal},
                        derived_from=[(column, table)],
                    )

                    return result, expr

        if {Attribute.numeric, Attribute.not_null} <= column.attributes:
            if set(sample.get_column(column.name).unique()) == {0, 1}:
                result = Column(
                    name=column.name,
                    attributes={Attribute.boolean},
                    derived_from=[(column, table)],
                )

                return result, column.expr.cast(polars.Boolean)

        return column, column.expr
