import dataclasses
import datetime
from typing import Iterable

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
                column.expr >= column.expr.max() - self.period
                for column in table.columns
                if column.data_type.is_temporal()
            ]

            if predicates:
                yield table.apply(lambda df: df.filter(predicates))
