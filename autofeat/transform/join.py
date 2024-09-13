import dataclasses
import itertools
import functools
from typing import Iterable, Set

import networkx
import polars
import polars._typing

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
    how: polars._typing.JoinStrategy = "outer"

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for related_tables in self._related_tables(list(tables)):
            if len(related_tables) > 1:
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
    ) -> Iterable[list[Table]]:
        graph = networkx.Graph()

        for i in range(len(tables)):
            graph.add_node(i)

        for x, y in itertools.combinations(range(len(tables)), 2):
            if self.on & tables[x].columns & tables[y].columns:
                graph.add_edge(x, y)

        for component in list(networkx.connected_components(graph)):
            yield [tables[i] for i in component]
