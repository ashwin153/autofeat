from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from autofeat.convert import IntoDataFrame, into_data_frame
from autofeat.transform import Extract

if TYPE_CHECKING:
    import polars

    from autofeat.convert import IntoDataFrame
    from autofeat.table import Table
    from autofeat.transform.base import Transform


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    """A collection of tables.

    :param tables: Tables in this dataset.
    """

    tables: list[Table]

    def apply(
        self,
        transform: Transform,
        /,
    ) -> Dataset:
        """Apply the ``transform`` to each table in this dataset.

        :param transform: Transform to apply.
        :return: Transformed dataset.
        """
        return Dataset(list(transform.apply(self.tables)))

    def features(
        self,
        known: IntoDataFrame,
    ) -> polars.DataFrame:
        """Extract features from all tables in this dataset that are relevant to the ``known`` data.

        .. note::

            Feature extraction is a computationally expensive operation.

        :param known: Data that is already known.
        :return: Extracted features.
        """
        return into_data_frame(self.apply(Extract(known=known)))

    def table(
        self,
        name: str,
    ) -> Table:
        """Get the table with the corresponding name.

        :param name: Name of the table.
        :return: Corresponding table.
        """
        for table in self.tables:
            if table.name == name:
                return table

        raise ValueError(f"table `{name}` does not exist")
