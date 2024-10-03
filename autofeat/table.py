from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import polars

if TYPE_CHECKING:

    from autofeat.attribute import Attribute


@dataclasses.dataclass(frozen=True, kw_only=True)
class Column:
    """

    :param attributes:
    :param derived_from:
    :param name:
    """

    attributes: set[Attribute] = dataclasses.field(default_factory=set)
    derived_from: list[tuple[Column, Table]] = dataclasses.field(default_factory=list)
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name

    @functools.cached_property
    def expr(
        self,
    ) -> polars.Expr:
        """

        :return:
        """
        return polars.col(self.name)

    def is_related(
        self,
        other: Column,
        /,
    ) -> bool:
        """Whether or not the columns are derived from a common ancestor.

        :param other: Other column.
        :return: Whether or not the columns share a common ancestor.
        """
        return not self._ancestors.isdisjoint(other._ancestors)

    @functools.cached_property
    def _ancestors(
        self,
    ) -> set[tuple[str, str]]:
        return {
            ancestor
            for parent, table in self.derived_from
            for ancestor in ((parent.name, table.name), *parent._ancestors)
        }


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A lazily-loaded data table.

    :param data: Contents of this table.
    :param name: Name of this table.
    :param schema: Structure of this table.
    """

    columns: list[Column]
    data: polars.LazyFrame = dataclasses.field(compare=False)
    name: str

    def __str__(
        self,
    ) -> str:
        return self.name

    def column(
        self,
        name: str,
    ) -> Column | None:
        """

        :param name:
        :return:
        """
        for column in self.columns:
            if column.name == name:
                return column

        return None

    def select(
        self,
        columns: list[Column],
    ) -> Table:
        """

        :return:
        """
        return Table(
            columns=columns,
            data=self.data.select(column.name for column in columns),
            name=self.name,
        )
