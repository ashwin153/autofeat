import itertools
from collections.abc import Iterable

import attrs
import networkx
import polars

from autofeat.table import Column, Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Union(Transform):
    """Vertically stack tables that have the same columns."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for related_tables in self._related_tables(list(tables)):
            if len(related_tables) == 1:
                yield related_tables[0]
            else:
                columns = [
                    Column(
                        name=column.name,
                        attributes=set.intersection(
                            *[
                                c.attributes
                                for t in related_tables
                                for c in t.columns
                                if c.name == column.name
                            ],
                        ),
                        derived_from=[
                            (c, t)
                            for t in related_tables
                            for c in t.columns
                            if c.name == column.name
                        ],
                    )
                    for column in related_tables[0].columns
                ]

                yield Table(
                    name=f"union({', '.join([t.name for t in related_tables])})",
                    data=polars.concat([t.data for t in related_tables], how="vertical"),
                    columns=columns,
                )

    def _related_tables(
        self,
        tables: list[Table],
    ) -> list[list[Table]]:
        graph: networkx.Graph[Table] = networkx.Graph()

        for table in tables:
            graph.add_node(table)

        for x, y in itertools.combinations(tables, 2):
            if {c.name for c in x.columns} == {c.name for c in y.columns}:
                graph.add_edge(x, y)

        return list(networkx.connected_components(graph))
