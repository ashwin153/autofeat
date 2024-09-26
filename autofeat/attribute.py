import enum


@enum.unique
class Attribute(enum.Enum):
    """A characteristic of a column."""

    boolean = enum.auto()
    """Contains ``True`` and ``False`` values."""

    categorical = enum.auto()
    """Contains categories."""

    not_null = enum.auto()
    """Has no missing values."""

    numeric = enum.auto()
    """Contains numbers."""

    pivotable = enum.auto()
    """Has low cardinality."""

    primary_key = enum.auto()
    """Is a component of the primary key."""

    textual = enum.auto()
    """Contains strings."""
