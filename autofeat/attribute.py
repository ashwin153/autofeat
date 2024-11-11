from __future__ import annotations

import enum

import polars


@enum.unique
class Attribute(enum.Enum):
    """A characteristic of a column."""

    aggregable = enum.auto()
    """Contains values that can be aggregated."""

    boolean = enum.auto()
    """Contains ``True`` and ``False`` values."""

    categorical = enum.auto()
    """Contains categories."""

    embedding = enum.auto()
    """Contains text embeddings."""

    not_null = enum.auto()
    """Has no missing values."""

    numeric = enum.auto()
    """Contains numbers."""

    primary_key = enum.auto()
    """Is a component of the primary key."""

    temporal = enum.auto()
    """Contains dates, times, or datetimes."""

    textual = enum.auto()
    """Contains strings."""

    @staticmethod
    def infer(
        *,
        data_type: polars.DataType,
        len: int,
        n_unique: int,
        null_count: int,
    ) -> set[Attribute]:
        """Infer attributes from column metadata.

        :param data_type: Type of data.
        :param len: Number of rows.
        :param n_unique: Number of unique values.
        :param null_count: Number of null values.
        :return: Inferred attributes.
        """
        attributes = set()

        if data_type.is_numeric():
            attributes.add(Attribute.aggregable)

        if isinstance(data_type, polars.Boolean):
            attributes.add(Attribute.boolean)

        if n_unique <= 50:
            attributes.add(Attribute.categorical)

        if null_count == 0:
            attributes.add(Attribute.not_null)

        if data_type.is_numeric():
            attributes.add(Attribute.numeric)

        if n_unique == len:
            attributes.add(Attribute.primary_key)

        if data_type.is_temporal():
            attributes.add(Attribute.temporal)

        if isinstance(data_type, polars.String):
            attributes.add(Attribute.textual)

        return attributes
