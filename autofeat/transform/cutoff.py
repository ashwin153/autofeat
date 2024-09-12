import datetime
from typing import Iterable

import polars

from autofeat.table import Table
from autofeat.transform import Transform


class Cutoff(Transform):
    """Filter tables to rows with timestamps within the ``[min, max]`` interval.

    :param max: Latest timestamp that rows must have occurred on or before.
    :param min: Earliest timestamp that rows must have occurred on or after.
    """

    max: datetime.datetime | None = None
    min: datetime.datetime | None = None

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for table in tables:
            predicates = []

            for column, data_type in table.schema.items():
                expr = polars.col(column)

                if self.max:
                    if isinstance(data_type, polars.Datetime):
                        predicates.append(expr.filter(expr <= self.max))
                    elif isinstance(data_type, polars.Date):
                        predicates.append(expr.filter(expr <= self.max.date()))
                    elif isinstance(data_type, polars.Time):
                        predicates.append(expr.filter(expr <= self.max.time()))

                if self.min:
                    if isinstance(data_type, polars.Datetime):
                        predicates.append(expr.filter(expr >= self.min))
                    elif isinstance(data_type, polars.Date):
                        predicates.append(expr.filter(expr >= self.min.date()))
                    elif isinstance(data_type, polars.Time):
                        predicates.append(expr.filter(expr >= self.min.time()))

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
            else:
                yield table
