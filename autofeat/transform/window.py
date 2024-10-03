import dataclasses
import datetime
from collections.abc import Iterable

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Window(Transform):
    """Filter out rows from tables that are older than each of the ``periods``.

    :param periods: Window sizes.
    """

    periods: list[datetime.timedelta]

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        now = datetime.datetime.now(datetime.UTC)

        for table in tables:
            temporal_columns = [
                column
                for column in table.columns
                if Attribute.temporal in column.attributes
            ]

            for period in self.periods:
                for temporal_column in temporal_columns:
                    yield Table(
                        data=table.data.filter(temporal_column.expr >= now - period),
                        name=f"window({table.name}, {period})",
                        columns=table.columns,
                    )
