import dataclasses
from typing import Iterable

import polars

from autofeat.table import Table
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
                for column in table.schema
            ]

            yield table.apply(lambda df: df.select(columns))

    def _cast(
        self,
        table: Table,
        column: str,
    ) -> polars.Expr:
        if isinstance(table.schema[column], polars.String):
            date = polars.col(column).str.to_date()
            if table.is_valid(date):
                return date

            datetime = polars.col(column).str.to_datetime()
            if table.is_valid(datetime):
                return datetime

            time = polars.col(column).str.to_time()
            if table.is_valid(time):
                return time

            category_sizes = table.sample.get_column(column).unique_counts()
            if max(category_sizes) > 1:
                return polars.col(column).cast(polars.Categorical)

        return polars.col(column)
