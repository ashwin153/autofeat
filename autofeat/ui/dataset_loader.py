import streamlit
import streamlit.runtime.caching.hashing

from autofeat import Dataset, source


def dataset_loader(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    streamlit.title("Load Dataset")

    source_type = streamlit.selectbox(
        "Source Type",
        ["Kaggle", "CSV"],
    )

    if source_type == "CSV":
        files = streamlit.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
        )

        if files:
            return _load_dataset_from_csv([file.name for file in files])

    if source_type == "Kaggle":
        name = streamlit.text_input(
            "Dataset or Competition",
            placeholder="house-prices-advanced-regression-techniques",
        )

        if name:
            return _load_dataset_from_kaggle(name)

    return None


@streamlit.cache_resource
def _load_dataset_from_csv(
    files: list[str],
) -> Dataset:
    return source.from_csv(files)


@streamlit.cache_resource
def _load_dataset_from_kaggle(
    name: str,
) -> Dataset:
    return source.from_kaggle(name)
