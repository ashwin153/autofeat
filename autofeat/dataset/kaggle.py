import dataclasses
import tempfile
import os
import polars
from typing import Iterable

from kaggle.api.kaggle_api_extended import KaggleApi

from dataset.base import Dataset, Table


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleDataset(Dataset):
    """A dataset stored in Kaggle.

    >>> KaggleDataset(id="abdullah0a/urban-air-quality-and-health-impact-dataset")

    .. code-block:: bash

        mkdir -p ~/.config/kaggle
        echo '{"username":"$KAGGLE_USERNAME", "key":"$KAGGLE_KEY"}' >~/.config/kaggle/kaggle.json
        chmod 600 ~/.config/kaggle/kaggle.json
    """

    id: str
    
    def list_tables(
        self,
    ) -> Iterable[Table]:
        api = KaggleApi()

        api.authenticate()

        path = api.get_default_download_dir(
            "datasets",
            *api.split_dataset_string(self.id),
        )

        api.dataset_download_files(
            dataset=self.id, 
            path=path, 
            unzip=True,
        )

        for file in os.listdir(path):
            yield Table(data=polars.scan_csv(os.path.join(path, file)))
