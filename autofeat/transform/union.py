import collections
import dataclasses
from collections.abc import Iterable

import polars

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Union(Transform):
    """Vertically stack tables that have the same columns."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for group in self._groups(list(tables)):
            if len(group) == 1:
                yield group[0]
            else:
                columns = [
                    Column(
                        name=column.name,
                        attributes=set.intersection(
                            *[
                                c.attributes
                                for t in group
                                for c in t.columns
                                if c.name == column.name
                            ],
                        ),
                        derived_from=[
                            (c, t)
                            for t in group
                            for c in t.columns
                            if c.name == column.name
                        ],
                    )
                    for column in group[0].columns
                ]

                yield Table(
                    name=f"union({', '.join([t.name for t in group])})",
                    data=polars.concat([t.data for t in group], how="vertical"),
                    columns=columns,
                )

    def _groups(
        self,
        tables: list[Table],
    ) -> list[list[Table]]:
        groups = collections.defaultdict(list)
        for table in tables:
            columns = frozenset(column.name for column in table.columns)
            groups[columns].append(table)

        return list(groups.values())
