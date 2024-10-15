from collections.abc import Iterable

import attrs

from autofeat.table import Table
from autofeat.transform.base import Transform


@attrs.define(frozen=True, kw_only=True, slots=True)
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
