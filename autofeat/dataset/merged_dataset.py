import dataclasses
from typing import Iterable

from autofeat.dataset.base import Dataset
from autofeat.table import Table


@dataclasses.dataclass(frozen=True)
class MergedDataset(Dataset):
    """A composition of multiple datasets.

    :param datasets: Component datasets.
    """

    datasets: list[Dataset]

    def tables(
        self,
    ) -> Iterable[Table]:
        yield from (
            table
            for dataset in self.datasets
            for table in dataset.tables()
        )
