from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from autofeat.table import Table


class Transform(abc.ABC):
    """A transformation over a collection of tables."""

    @abc.abstractmethod
    def apply(
        self,
        tables: Iterable[Table],
    ) -> Iterable[Table]:
        """Apply this transformation to the ``tables``.

        :param tables: Tables to transform.
        :return: Transformed tables.
        """

    def then(
        self,
        head: Transform,
        /,
        *tail: Transform,
    ) -> Transform:
        """Apply any of the transforms after applying this transform.

        >>> from autofeat.transform import *

        >>> transform = (
        ...     Rename({"old": "new"})
        ...     .then(Cast())
        ...     .then(Aggregate(by={"group"})
        ...     .then(Identity(), Join(on={"group"}).Combine())
        ... )

        :param head: Transform to apply after this transform.
        :param tail: Other transforms to apply after this transform.
        """
        from autofeat.transform.all_of import AllOf
        from autofeat.transform.any_of import AnyOf
        from autofeat.transform.identity import Identity

        transforms = []

        for next in (head, *tail):
            if isinstance(self, Identity):
                transforms.append(next)
            elif isinstance(next, Identity):
                transforms.append(self)
            else:
                transforms.append(
                    AllOf(
                        transforms=[
                            *(self.transforms if isinstance(self, AllOf) else [self]),
                            *(next.transforms if isinstance(next, AllOf) else [next]),
                        ],
                    ),
                )

        if len(transforms) == 1:
            return transforms[0]
        else:
            return AnyOf(transforms=transforms)
