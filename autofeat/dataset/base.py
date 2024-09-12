from __future__ import annotations

import abc
from typing import Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from autofeat.table import Table


class Dataset(abc.ABC):
    """A collection of tables."""

    @abc.abstractmethod
    def list_tables(
        self,
    ) -> Iterable[Table]:
        """List all tables in this dataset.

        :return: All tables.
        """
