import dataclasses
import os
import pathlib
import zipfile
from collections.abc import Iterable

import polars

from autofeat.dataset.base import Dataset
from autofeat.table import Table

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleDataset(Dataset):
    """A Kaggle dataset or competition.

    >>> dataset = KaggleDataset(id="abdullah0a/urban-air-quality-and-health-impact-dataset")

    >>> competition = KaggleDataset(id="house-prices-advanced-regression-techniques")

    .. tip::

        Create a `key <https://www.kaggle.com/settings>`_ from your account settings and save it to
        ``~/.config/kaggle/kaggle.json`` to provide credentials to the Kaggle API.

        .. code-block:: bash

            mkdir -p ~/.config/kaggle
            echo '{"username":"$USER", "key":"$KEY"}' >~/.config/kaggle/kaggle.json
            chmod 600 ~/.config/kaggle/kaggle.json

    :param cache: Path where datasets are locally cached.
    :param id: Name of the competition or dataset.
    :param sample_size: Number of rows to sample from each table in the dataset.
    """

    cache: pathlib.Path = DEFAULT_CACHE
    id: str
    sample_size: int = 10

    def tables(
        self,
    ) -> Iterable[Table]:
        path = self.cache / "datasets" / self.id
        path.mkdir(parents=True, exist_ok=True)

        if not any(path.iterdir()):
            archive = self._download(path)
            self._unzip(archive)

        for csv in path.glob("*.csv"):
            df = polars.read_csv(
                csv,
                null_values="NA",
            )

            yield Table(
                data=df.lazy(),
                name=csv.name,
                sample=df.sample(min(self.sample_size, len(df))),
            )

    def _download(
        self,
        path: pathlib.Path,
    ) -> pathlib.Path:
        # kaggle creates and authenticates a client on import
        import kaggle  # type: ignore[import-untyped]

        if "/" in self.id:
            kaggle.api.dataset_download_files(
                dataset=self.id,
                path=str(path),
            )

            return path / f"{kaggle.api.split_dataset_string(self.id)[1]}.zip"
        else:
            kaggle.api.competition_download_files(
                competition=self.id,
                path=str(path),
            )

            return path / f"{self.id}.zip"

    def _unzip(
        self,
        archive: pathlib.Path,
    ) -> None:
        with zipfile.ZipFile(archive) as zip:
            zip.extractall(str(archive.parent))

        os.remove(archive)
