import dataclasses
import pathlib
import polars
from typing import Iterable

from autofeat.dataset.base import Dataset
from autofeat.table import Table


DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


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

    :param cache: Path where datasets are locally cached.
    :param id: Dataset identifier in the form owner/name.
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
            df = polars.read_csv(csv)

            yield Table(
                data=polars.scan_csv(csv),
                name=csv.name,
                sample=df.sample(min(self.sample_size, len(df))),
            )