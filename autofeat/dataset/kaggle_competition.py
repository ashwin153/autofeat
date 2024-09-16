import dataclasses
from typing import Iterable

from autofeat.dataset.base import Dataset
from autofeat.table import Table


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleCompetition(Dataset):
    """A Kaggle competition."""

    def tables(
        self,
    ) -> Iterable[Table]:
        # TODO
        ...
