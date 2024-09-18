from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from autofeat.table import Table
    from autofeat.transform.base import Transform


class Dataset(abc.ABC):
    """A collection of tables."""

    @abc.abstractmethod
    def tables(
        self,
    ) -> Iterable[Table]:
        """List all tables in this dataset.

        :return: All tables.
        """

    def derive(
        self,
        transform: Transform,
        /,
    ) -> Dataset:
        """Apply the ``transform`` to each table in this dataset.

        :param transform: Transform to apply.
        :return: Derived dataset.
        """
        from autofeat.dataset.derived_dataset import DerivedDataset

        return DerivedDataset(dataset=self, transform=transform)

    def merge(
        self,
        *datasets: Dataset,
    ) -> Dataset:
        """Add the ``datasets`` to this dataset.

        :param datasets: Datasets to merge.
        :return: Merged dataset.
        """
        from autofeat.dataset.merged_dataset import MergedDataset

        if datasets:
            return MergedDataset(datasets=[self, *datasets])
        else:
            return self

    def table(
        self,
        name: str,
    ) -> Table:
        """Get the table with the corresponding name.

        :param name: Name of the table.
        :return: Corresponding table.
        """
        for table in self.tables():
            if table.name == name:
                return table

        raise ValueError(f"table `{name}` does not exist")
