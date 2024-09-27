import dataclasses
import itertools
from collections.abc import Iterable

import polars

from autofeat.attribute import Attribute
from autofeat.schema import Schema
from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Aggregate(Transform):
    """Group by a set of columns and aggregate the remaining columns in various ways.

    :param max_pivots: Maximum number of columns that can be pivoted at a time.
    """

    max_pivots: int = 1

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            if aggregations := dict(self._aggregations(table)):
                for primary_key in self._primary_keys(table):
                    schema = Schema({
                        **{
                            column: table.schema[column] | {Attribute.primary_key}
                            for column in primary_key
                        },
                        **{
                            column: {Attribute.numeric, Attribute.not_null}
                            for column in aggregations
                        },
                    })

                    yield Table(
                        name=f"group_by({table.name}, {', '.join(sorted(primary_key))})",
                        schema=schema,
                        data=table.data.group_by(primary_key).agg(**aggregations),
                    )

    def _primary_keys(
        self,
        table: Table,
    ) -> Iterable[tuple[str, ...]]:
        pivotable_columns = table.schema.select(
            include={Attribute.pivotable},
            exclude={Attribute.primary_key},
        )

        for count in range(1, self.max_pivots + 1):
            yield from itertools.combinations(pivotable_columns, count)

    def _aggregations(
        self,
        table: Table,
    ) -> Iterable[tuple[str, polars.Expr]]:
        aggregable_columns = table.schema.select(
            include={Attribute.numeric},
        )

        yield "count(*)", polars.count()

        for column in aggregable_columns:
            yield f"max({column})", polars.col(column).max()
            yield f"mean({column})", polars.col(column).mean()
            yield f"median({column})", polars.col(column).median()
            yield f"min({column})", polars.col(column).min()
            yield f"std({column})", polars.col(column).std()
            yield f"sum({column})", polars.col(column).sum()
            yield f"var({column})", polars.col(column).var()
