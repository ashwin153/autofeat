from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars

    from autofeat.schema import Schema


@dataclasses.dataclass(frozen=True, kw_only=True)
class Table:
    """A lazily-loaded data table.

    :param data: Contents of this table.
    :param name: Name of this table.
    :param schema: Structure of this table.
    """

    data: polars.LazyFrame
    name: str
    schema: Schema
