import os
import pathlib
import shutil
import zipfile

import polars

from autofeat.dataset import Dataset
from autofeat.table import Table

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


def from_kaggle(
    name: str,
    *,
    cache: pathlib.Path = DEFAULT_CACHE,
    sample_size: int = 10,
) -> Dataset:
    """Source tables from the Kaggle dataset or competition with the ``name``.

    .. note::

        Dataset names are prefixed by the name of the owner, but competition names are not.

        >>> dataset = KaggleDataset(name="abdullah0a/urban-air-quality-and-health-impact-dataset")
        >>> competition = KaggleDataset(name="house-prices-advanced-regression-techniques")

    .. tip::

        Create a `key <https://www.kaggle.com/settings>`_ from your account settings and save it to
        ``~/.config/kaggle/kaggle.json`` to provide credentials to the Kaggle API.

        .. code-block:: bash

            mkdir -p ~/.config/kaggle
            echo '{"username":"$USER", "key":"$KEY"}' >~/.config/kaggle/kaggle.json
            chmod 600 ~/.config/kaggle/kaggle.json

    :param name: Name of the competition or dataset.
    :param cache: Path where datasets are locally cached.
    :param sample_size: Number of rows to sample from each table in the dataset.
    :return: Kaggle tables.
    """
    path = cache / "data" / name

    path.mkdir(parents=True, exist_ok=True)

    if not any(path.iterdir()):
        try:
            # kaggle creates and authenticates a client on import
            import kaggle  # type: ignore[import-untyped]

            if "/" in name:
                kaggle.api.dataset_download_files(name, str(path))
                archive = path / f"{kaggle.api.split_dataset_string(name)[1]}.zip"
            else:
                kaggle.api.competition_download_files(name, str(path))
                archive = path / f"{name}.zip"

            with zipfile.ZipFile(archive) as zip:
                zip.extractall(str(archive.parent))

            os.remove(archive)
        except Exception:
            shutil.rmtree(path)
            raise

    tables = []

    for csv in path.glob("*.csv"):
        df = polars.read_csv(csv, null_values="NA")

        table = Table(
            data=df.lazy(),
            name=csv.name,
            sample=df.sample(min(sample_size, len(df))),
        )

        tables.append(table)

    return Dataset(tables)
