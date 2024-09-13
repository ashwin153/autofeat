import dataclasses
import datetime
from typing import Iterable

import polars

from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Window(Transform):
    """Filter out rows from tables that are older than the period.

    :param period: Size of the window.
    """

    period: datetime.timedelta

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            predicates = [
                polars.col(column) >= polars.col(column).max() - self.period
                for column, data_type in table.schema.items()
                if data_type.is_temporal()
            ]

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
