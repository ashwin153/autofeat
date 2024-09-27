from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

from autofeat.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from autofeat.attribute import Attribute


IntoSchema: TypeAlias = Union[
    "Schema",
    "Mapping[str, Collection[Attribute]]",
]


def into_schema(
    *values: IntoSchema,
) -> IntoSchema:
    """Convert the ``values`` into a schema.

    :param values: Values to convert.
    :return: Converted schema.
    """
    return Schema(dict(_columns(*values)))


def _columns(
    *values: IntoSchema,
) -> Iterable[tuple[str, set[Attribute]]]:
    from autofeat.schema import Schema

    for value in values:
        if isinstance(value, Schema):
            yield from value.items()
        elif isinstance(value, dict):
            yield from ((c, set(a)) for c, a in value.items())
        else:
            raise NotImplementedError(f"cannot extract columns from `{type(value)}`")
