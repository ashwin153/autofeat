import dataclasses
from typing import Iterable

from autofeat.table import Table
from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class Identity(Transform):
    """A transform that performs no table modifications.

    .. tip::

        Used to passthrough the base transform in :meth:`~Transform.then`.
    """

    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        return tables
