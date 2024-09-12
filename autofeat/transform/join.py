import dataclasses
import itertools
import functools
from typing import Iterable, Set

import networkx
import polars

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Join(Transform):
    """Join tables on common key columns.

    .. tip::

        Aggregate tables by the same key columns that they are joined on to avoid join explosion.

    :param on: Columns to join on.
    :param how: Method of joining the tables.
    """

    on: Set[str]
    how: polars.JoinStrategy = "outer"

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for related_tables in self._related_tables(list(tables)):
            yield functools.reduce(
                lambda x, y: x.join(
                    y,
                    on=list(self.on & x.columns & y.columns),
                    how=self.how,
                ),
                related_tables,
            )

    def _related_tables(
        self,
        tables: list[Table],
    ) -> list[list[Table]]:
        graph = networkx.Graph()

        for x in tables:
            graph.add_node(x)

        for x, y in itertools.combinations(tables, 2):
            if self.on & x.columns & y.columns:
                graph.add_edge(x, y)

        return list(networkx.connected_components(graph))
