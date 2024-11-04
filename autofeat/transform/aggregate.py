import dataclasses
import datetime
import itertools
from collections.abc import Collection, Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.convert.into_exprs import into_exprs
from autofeat.convert.into_named_exprs import into_named_exprs
from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Aggregate(Transform):
    """Group by a set of columns and aggregate the remaining columns in various ways.

    :param is_pivotable: Columns that are allowed to be pivoted.
    :param max_pivots: Maximum number of columns that can be pivoted at a time.
    :param windows: Time windows over which to aggregate data.
    """

    is_pivotable: Collection[str | Column]
    max_pivots: int = 1
    windows: list[datetime.timedelta] | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            pivotable_columns = [
                *self._pivotable_columns(table),
            ]

            temporal_columns = [
                *self._temporal_columns(table),
            ]

            aggregations = [
                *self._aggregations(table, pivotable_columns),
            ]

            if aggregations and pivotable_columns:
                pivoted_columns = [
                    column
                    for num_pivots in range(1, self.max_pivots + 1)
                    for pivot in itertools.combinations(pivotable_columns, num_pivots)
                    for column in pivot
                ]

                yield Table(
                    columns=[
                        *pivoted_columns,
                        *[column for column, _ in aggregations],
                    ],
                    data=(
                        table.data
                        .group_by(into_exprs(pivoted_columns))
                        .agg(**into_named_exprs(aggregations))
                    ),
                    name="group_by({table}, {pivot})".format(
                        table=table.name,
                        pivot=", ".join(str(column) for column in pivoted_columns),
                    ),
                )

                if self.windows and temporal_columns:
                    for temporal_column in temporal_columns:
                        for window in self.windows:
                            yield Table(
                                columns=[
                                    temporal_column,
                                    *pivoted_columns,
                                    *[column for column, _ in aggregations],
                                ],
                                data=(
                                    table.data
                                    .group_by_dynamic(
                                        temporal_column.name,
                                        every=window,
                                        group_by=into_exprs(pivoted_columns),
                                    )
                                    .agg(**into_named_exprs(aggregations))
                                ),
                                name="group_by({table}, {pivot}, {window})".format(
                                    table=table.name,
                                    pivot=", ".join(str(column) for column in pivoted_columns),
                                    window=str(window),
                                ),
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

    def _pivotable_columns(
        self,
        table: Table,
    ) -> Iterable[Column]:
        return [
            Column(
                name=column.name,
                attributes=column.attributes | {Attribute.primary_key},
                derived_from=[(column, table)],
            )
            for column in table.columns
            if Attribute.primary_key not in column.attributes
            if any(column.name == str(c) for c in self.is_pivotable)
        ]

    def _temporal_columns(
        self,
        table: Table,
    ) -> Iterable[Column]:
        return [
            column
            for column in table.columns
            if Attribute.temporal in column.attributes
        ]
