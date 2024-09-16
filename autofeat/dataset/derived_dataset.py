from typing import Iterable

import dataclasses

from autofeat.dataset.base import Dataset
from autofeat.table import Table
from autofeat.transform import Transform


@dataclasses.dataclass(frozen=True, kw_only=True)
class DerivedDataset(Dataset):
    """A transformation of a dataset.

    :param dataset: Dataset to transform.
    :param transform: Transform to apply.
    """

    dataset: Dataset
    transform: Transform

    def tables(
        self,
    ) -> Iterable[Table]:
        return self.transform.apply(self.dataset.tables())