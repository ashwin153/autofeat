import pandas
import pygwalker.api.streamlit
import streamlit

from autofeat import Dataset


def dataset_explorer(
    dataset: Dataset,
) -> None:
    """Explore a sample of the ``dataset``.

    :param dataset: Dataset to explore.
    """
    streamlit.header("Explore Dataset")

    sample_size = streamlit.number_input(
        "Sample Size",
        min_value=1,
        max_value=10000,
        value=100,
        step=50,
    )

    for tab, sample in zip(
        streamlit.tabs([table.name for table in dataset.tables]),
        _load_samples(dataset, sample_size),
    ):
        with tab:
            renderer = pygwalker.api.streamlit.StreamlitRenderer(
                sample,
                default_tab="data",
            )

            renderer.explorer()


@streamlit.cache_data(
    hash_funcs={Dataset: id},
    max_entries=1,
)
def _load_samples(
    dataset: Dataset,
    sample_size: int,
) -> list[pandas.DataFrame]:
    return [
        table.data.head(sample_size).collect().to_pandas()
        for table in dataset.tables
    ]
