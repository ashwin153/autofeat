import dataclasses
import functools
import itertools
from collections.abc import Iterable

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

    on: set[str]
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
                        on=self._join_columns(x, y),
                        how=self.how,
                    ),
                    related_tables,
                )

    def _join_columns(
        self,
        x: Table,
        y: Table,
    ) -> list[str]:
        x_columns = {column.name for column in x.columns}
        y_columns = {column.name for column in y.columns}
        return list(self.on & x_columns & y_columns)

    def _related_tables(
        self,
        tables: list[Table],
    ) -> Iterable[list[Table]]:
        graph: networkx.Graph[int] = networkx.Graph()

        for i in range(len(tables)):
            graph.add_node(i)

        for x, y in itertools.combinations(range(len(tables)), 2):
            if self._join_columns(tables[x], tables[y]):
                graph.add_edge(x, y)

        for component in list(networkx.connected_components(graph)):
            yield [tables[i] for i in component]
