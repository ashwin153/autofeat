import pygwalker.api.streamlit
import streamlit

from autofeat import Dataset, Table
from autofeat.settings import SETTINGS


def explore_dataset(
    dataset: Dataset,
) -> None:
    """Explore a sample of the ``dataset``.

    :param dataset: Dataset to explore.
    """
    with streamlit.expander("Explore Dataset"):
        with streamlit.form("explore_dataset"):
            table = streamlit.selectbox(
                help="Table to explore",
                label="Table",
                options=dataset.tables,
            )

            sample_size = streamlit.number_input(
                "Sample Size",
                help="Number of rows to load",
                min_value=1,
                max_value=10000,
                value=100,
                step=50,
            )

            load = streamlit.form_submit_button("Load")

        if load:
            sample = _render_sample(table, sample_size)
            sample.table(key=table.name)


@streamlit.cache_resource(
    hash_funcs={
        Table: id,
    },
    max_entries=20,
)
def _render_sample(  # type: ignore[no-any-unimported]
    table: Table,
    sample_size: int,
) -> pygwalker.api.streamlit.StreamlitRenderer:
    return pygwalker.api.streamlit.StreamlitRenderer(
        table.data.head(sample_size).collect(),
        appearance="dark" if SETTINGS.dark_mode else "light",
    )
