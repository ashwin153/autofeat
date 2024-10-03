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

    is_pivotable: Collection[str | Column] | None = None
    max_pivots: int = 1

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if aggregations := list(self._aggregations(table)):
                for pivoted_columns in list(self._pivoted_columns(table)):
                    columns = [
                        *pivoted_columns,
                        *[column for column, _ in aggregations],
                    ]

                    data = (
                        table.data
                        .group_by(into_exprs(pivoted_columns))
                        .agg(**into_named_exprs(aggregations))
                    )

                    yield Table(
                        columns=columns,
                        data=data,
                        name=f"group_by({table.name}, {pivoted_columns})",
                    )

    def _pivoted_columns(
        self,
        table: Table,
    ) -> Iterable[tuple[Column, ...]]:
        is_pivotable = {
            str(column)
            for column in self.is_pivotable or []
        }

        pivotable_columns = [
            Column(
                name=column.name,
                attributes=column.attributes | {Attribute.primary_key},
                derived_from=[(column, table)],
            )
            for column in table.columns
            if Attribute.pivotable in column.attributes
            if Attribute.primary_key not in column.attributes
            if not is_pivotable or column.name in is_pivotable
        ]

        for count in range(1, self.max_pivots + 1):
            yield from itertools.combinations(pivotable_columns, count)

    def _aggregations(
        self,
        table: Table,
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
            if Attribute.pivotable not in column.attributes
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
