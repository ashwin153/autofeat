import dataclasses
import pathlib
import polars
from typing import Iterable

import kaggle

from autofeat.dataset.base import Dataset
from autofeat.transform import Table


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleDataset(Dataset):
    """A dataset stored in Kaggle.

    >>> KaggleDataset(id="abdullah0a/urban-air-quality-and-health-impact-dataset")

    .. code-block:: bash

        mkdir -p ~/.config/kaggle
        echo '{"username":"$KAGGLE_USERNAME", "key":"$KAGGLE_KEY"}' >~/.config/kaggle/kaggle.json
        chmod 600 ~/.config/kaggle/kaggle.json

    .. tip::

        Create a `key <https://www.kaggle.com/settings>`_ from your account settings.

    :param id:
    :param sample_size:
    """

    id: str
    cache: pathlib.Path = pathlib.Path.home() / "cache" / "kaggle"
    sample_size: int = 250

    def list_tables(
        self,
    ) -> Iterable[Table]:
        api = kaggle.KaggleApi()

        api.authenticate()

        path = self.cache / "datasets" / self.id

        api.dataset_download_files(
            dataset=self.id,
            path=str(path),
            unzip=True,
        )

        for csv in path.iterdir():
            yield Table(
                data=polars.scan_csv(csv),
                sample=polars.read_csv(csv).sample(self.sample_size),
            )
