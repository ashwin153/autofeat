import dataclasses
import pathlib
import polars
from typing import Iterable

from autofeat.dataset.base import Dataset
from autofeat.table import Table


DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleCompetition(Dataset):
    """A Kaggle competition."""

    def list_tables(
        self,
    ) -> Iterable[Table]:
        # TODO
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class KaggleDataset(Dataset):
    """A Kaggle dataset.

    >>> dataset = KaggleDataset(id="abdullah0a/urban-air-quality-and-health-impact-dataset")

    .. tip::

        Create a `key <https://www.kaggle.com/settings>`_ from your account settings and save it to
        ``~/.config/kaggle/kaggle.json`` to provide credentials to the Kaggle API.

        .. code-block:: bash

            mkdir -p ~/.config/kaggle
            echo '{"username":"$USER", "key":"$KEY"}' >~/.config/kaggle/kaggle.json
            chmod 600 ~/.config/kaggle/kaggle.json

    :param id: Dataset identifier in the form owner/name.
    :param cache: Path where datasets are locally cached.
    :param sample_size: Number of rows to sample from each table in the dataset.
    """

    id: str
    cache: pathlib.Path = DEFAULT_CACHE
    sample_size: int = 250

    def list_tables(
        self,
    ) -> Iterable[Table]:
        path = self.cache / "datasets" / self.id

        # TODO: also redownload when the dataset version changes
        if not any(path.iterdir()):
            # kaggle creates and authenticates a client on import
            import kaggle

            kaggle.api.dataset_download_files(
                dataset=self.id,
                path=str(path),
                unzip=True,
            )

        for csv in path.glob("*.csv"):
            yield Table(
                data=polars.scan_csv(csv),
                sample=polars.read_csv(csv).sample(self.sample_size),
            )
