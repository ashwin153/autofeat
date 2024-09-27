import contextlib
import os
import pathlib
import shutil
import zipfile

from autofeat.dataset import Dataset
from autofeat.source.from_csv import from_csv

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "kaggle"


def from_kaggle(
    name: str,
    *,
    cache: pathlib.Path = DEFAULT_CACHE,
) -> Dataset:
    """Load from Kaggle.

    .. note::

        Dataset names are prefixed by the name of the owner, but competition names are not.

        .. code-block:: python

            dataset = from_kaggle(
                name="abdullah0a/urban-air-quality-and-health-impact-dataset",
            )

            competition = from_kaggle(
                name="house-prices-advanced-regression-techniques",
            )

    .. tip::

        Create a `key <https://www.kaggle.com/settings>`_ from your account settings and save it to
        ``~/.config/kaggle/kaggle.json`` to provide credentials to the Kaggle API.

        .. code-block:: bash

            mkdir -p ~/.config/kaggle
            echo '{"username":"$USER", "key":"$KEY"}' >~/.config/kaggle/kaggle.json
            chmod 600 ~/.config/kaggle/kaggle.json

    :param name: Name of the competition or dataset.
    :param cache: Path where data is locally cached.
    :return: Dataset.
    """
    path = cache / "data" / name

    path.mkdir(parents=True, exist_ok=True)

    if not any(path.iterdir()):
        try:
            # kaggle creates and authenticates a client on import
            import kaggle  # type: ignore[import-untyped]

            with contextlib.redirect_stdout(None):
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

    return from_csv(
        path.glob("*.csv"),
        ignore_errors=True,
        null_values=["NA"],
    )
