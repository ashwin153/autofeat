import functools
import itertools
from collections.abc import Iterable

import attrs
import networkx

from autofeat.attribute import Attribute
from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
class Join(Transform):
    """Join tables on common primary key columns."""

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        for related_tables in self._related_tables(list(tables)):
            if len(related_tables) == 1:
                yield related_tables[0]
            else:
                yield functools.reduce(
                    lambda x, y: Table(
                        name=f"{x.name}, {y.name}",
                        data=x.data.join(
                            y.data,
                            how="outer",
                            on=self._key_columns(x, y),
                        ),
                        columns=[
                            *x.columns,
                            *[
                                y_column
                                for y_column in y.columns
                                if all(y_column.name != x_column.name for x_column in x.columns)
                            ],
                        ],
                    ),
                    related_tables,
                )

    def _key_columns(
        self,
        x: Table,
        y: Table,
    ) -> list[str]:
        x_columns = {
            column.name
            for column in x.columns
        }

        x_primary_key = {
            column.name
            for column in x.columns
            if Attribute.primary_key in column.attributes
        }

        y_columns = {
            column.name
            for column in y.columns
        }

        y_primary_key = {
            column.name
            for column in y.columns
            if Attribute.primary_key in column.attributes
        }

        if x_primary_key and x_primary_key <= y_columns:
            return list(x_primary_key)
        elif y_primary_key and y_primary_key <= x_columns:
            return list(y_primary_key)
        else:
            return []

    def _related_tables(
        self,
        tables: list[Table],
    ) -> list[list[Table]]:
        graph: networkx.Graph[Table] = networkx.Graph()

        for table in tables:
            graph.add_node(table)

        for x, y in itertools.combinations(tables, 2):
            if self._key_columns(x, y):
                graph.add_edge(x, y)

        return list(networkx.connected_components(graph))
