import dataclasses
from collections.abc import Iterable

import polars

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cast(Transform):
    """Cast columns to more appropriate types.

    In particular, string columns containing date, time, or datetime values are converted to
    temporal columns and string columns containing duplicate rows are converted to categorical
    columns. Casting enables further transformation (e.g., time columns enable rolling aggregation)
    and improves performance (e.g., categorical columns are easier to join).
    """

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            columns = [
                self._cast(table, column)
                for column in table.columns
            ]

            yield table.apply(lambda df: df.select(columns))

    def _cast(
        self,
        table: Table,
        column: Column,
    ) -> polars.Expr:
        if (
            isinstance(column.data_type, polars.String)
            and not table.sample.select(column.expr.is_null().all()).item()
        ):
            date = column.expr.str.to_date("%Y-%m-%d")
            if table.is_valid(date):
                return date

            datetime = column.expr.str.to_datetime()
            if table.is_valid(datetime):
                return datetime

            time = column.expr.str.to_time()
            if table.is_valid(time):
                return time

            largest_category = table.sample.select(column.expr.unique_counts().max()).item()
            if largest_category > 1:
                return column.expr.cast(polars.Categorical)

        return column.expr
