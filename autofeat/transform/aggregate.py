import dataclasses
import itertools
from collections.abc import Collection, Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert import into_exprs, into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Aggregate(Transform):
    """Group by a set of columns and aggregate the remaining columns in various ways.

    :param allowed_pivots: Columns that are allowed to be pivoted.
    :param max_pivots: Maximum number of columns that can be pivoted at a time.
    """

    is_pivotable: Collection[str | Column]
    max_pivots: int = 1

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            pivotable_columns = [
                Column(
                    name=column.name,
                    attributes=column.attributes | {Attribute.primary_key},
                    derived_from=[(column, table)],
                )
                for column in table.columns
                if Attribute.primary_key not in column.attributes
                if any(column.name == str(c) for c in self.is_pivotable)
            ]

            aggregations = [
                *self._aggregations(table, pivotable_columns),
            ]

            if aggregations and pivotable_columns:
                for num_pivots in range(1, self.max_pivots + 1):
                    for pivots in itertools.combinations(pivotable_columns, num_pivots):
                        columns = [
                            *pivots,
                            *[column for column, _ in aggregations],
                        ]

                        data = (
                            table.data
                            .group_by(into_exprs(pivots))
                            .agg(**into_named_exprs(aggregations))
                        )

                        yield Table(
                            columns=columns,
                            data=data,
                            name=f"group_by({table.name}, {', '.join(str(c) for c in pivots)})",
                        )

    def _aggregations(
        self,
        table: Table,
        pivotable_columns: list[Column],
    ) -> Iterable[tuple[Column, polars.Expr]]:
        yield (
            Column(name="count(*)", attributes={Attribute.numeric, Attribute.not_null}),
            polars.count(),
        )

        numeric_columns = [
            column
            for column in table.columns
            if Attribute.numeric in column.attributes
            if Attribute.primary_key not in column.attributes
            if all(column.name != c.name for c in pivotable_columns)
        ]

        for x in numeric_columns:
            aggregations = [
                (f"max({x})", x.expr.max()),
                (f"mean({x})", x.expr.mean()),
                (f"median({x})", x.expr.median()),
                (f"min({x})", x.expr.min()),
                (f"std({x})", x.expr.std()),
                (f"sum({x})", x.expr.sum()),
                (f"var({x})", x.expr.var()),
            ]

            for name, expr in aggregations:
                column = Column(
                    name=name,
                    attributes=x.attributes | {Attribute.not_null},
                    derived_from=[(x, table)],
                )

                yield column, expr
