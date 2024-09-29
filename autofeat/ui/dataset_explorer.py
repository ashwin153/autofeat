import pandas
import pygwalker.api.streamlit
import streamlit
import streamlit.runtime.caching.hashing

from autofeat import Dataset


def dataset_explorer(
    dataset: Dataset,
) -> None:
    """Explore a sample of the ``dataset``.

    :param dataset: Dataset to explore.
    """
    streamlit.title("Explore Dataset")

    sample_size = streamlit.number_input(
        "Sample Size",
        min_value=1,
        max_value=10000,
        value=100,
    )

    for tab, sample in zip(
        streamlit.tabs([table.name for table in dataset.tables]),
        _load_samples(dataset, sample_size),
    ):
        with tab:
            renderer = pygwalker.api.streamlit.StreamlitRenderer(sample)
            renderer.explorer()


@streamlit.cache_data(hash_funcs={Dataset: id})
def _load_samples(
    dataset: Dataset,
    sample_size: int,
) -> list[pandas.DataFrame]:
    return [
        table.data.head(sample_size).collect().to_pandas()
        for table in dataset.tables
    ]
