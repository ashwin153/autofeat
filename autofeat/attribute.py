import enum


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
