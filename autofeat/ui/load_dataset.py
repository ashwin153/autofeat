import streamlit

from autofeat import Dataset, source
from autofeat.transform import Cast, Encode


def load_dataset(

) -> Dataset | None:
    """Load a dataset from a configurable source.

    :return: Loaded dataset.
    """
    source_type = streamlit.selectbox(
        "Source Type",
        ["Kaggle", "CSV"],
    )

    if source_type == "CSV":
        csv_files = streamlit.file_uploader(
            "Upload Files",
            accept_multiple_files=True,
            type="csv",
        )

        if csv_files:
            return _source_dataset_from_csv([file.name for file in csv_files])

    if source_type == "Kaggle":
        kaggle_name = streamlit.text_input(
            "Dataset / Competition",
            placeholder="house-prices-advanced-regression-techniques",
        )

        if kaggle_name:
            return _source_dataset_from_kaggle(kaggle_name)

    return None


@streamlit.cache_resource(
    max_entries=1,
)
def _source_dataset_from_csv(
    files: list[str],
) -> Dataset:
    return _clean_dataset(source.from_csv(files))


@streamlit.cache_resource(
    max_entries=1,
)
def _source_dataset_from_kaggle(
    name: str,
) -> Dataset:
    return _clean_dataset(source.from_kaggle(name))


def _clean_dataset(
    dataset: Dataset,
) -> Dataset:
    return dataset.apply(Cast().then(Encode()))
