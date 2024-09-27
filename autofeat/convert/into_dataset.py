from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from autofeat.dataset import Dataset
    from autofeat.table import Table


_IntoDataset: TypeAlias = Union[
    "Table",
    "Dataset",
]


IntoDataset: TypeAlias = Union[
    _IntoDataset,
    Iterable[_IntoDataset],
]


def into_dataset(
    *values: IntoDataset,
) -> Dataset:
    """Convert the ``values`` into a dataset.

    :param values: Values to convert.
    :return: Converted dataset.
    """
    from autofeat.dataset import Dataset

    return Dataset(list(_tables(*values)))


def _tables(
    *values: IntoDataset,
) -> Iterable[Table]:
    from autofeat.dataset import Dataset
    from autofeat.table import Table

    for value in values:
        if isinstance(value, Table):
            yield value
        elif isinstance(value, Dataset):
            yield from value.tables
        elif isinstance(value, Iterable):
            yield from (t for v in value for t in _tables(v))
        else:
            raise NotImplementedError(f"`{type(value)}` cannot be converted to a dataset")
