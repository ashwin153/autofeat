import dataclasses
import datetime
from collections.abc import Iterable

import polars

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
            for period in self.periods:
                for column in table.schema.select(include={Attribute.temporal}):
                    yield Table(
                        data=table.data.filter(polars.col(column) >= now - period),
                        name=f"window({table.name}, {period})",
                        schema=table.schema,
                    )
