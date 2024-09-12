import datetime
from typing import Iterable

import polars

from autofeat.table import Table
from autofeat.transform import Transform


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
            predicates = []

            for column, data_type in table.schema.items():
                if data_type.is_temporal():
                    expr = polars.col(column)
                    predicates.append(expr.filter(expr >= expr.max() - self.period))

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
