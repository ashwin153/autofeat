import streamlit

from autofeat.attribute import Attribute
from autofeat.dataset import Dataset
from autofeat.solver import SOLVERS, Problem
from autofeat.transform import Aggregate, Cast, Drop, Encode, Identity, Transform


def feature_loader(
    dataset: Dataset,
) -> None:
    """Load features from the ``dataset`` that are relevant to a prediction problem.

    :param dataset: Dataset to load features from.
    """
    streamlit.title("Load Features")

    table = streamlit.selectbox(
        "Table",
        dataset.tables,
        index=None,
        key="table",
        on_change=lambda: _clear_state("target_column", "known_columns", "problem", "solver"),
    )

    if not table:
        return

    target_column = streamlit.selectbox(
        "Target Column",
        table.schema,
        index=None,
        key="target_column",
        on_change=lambda: _clear_state("known_columns", "problem", "solver"),
    )

    if not target_column:
        return

    known_columns = streamlit.multiselect(
        "Known Columns",
        [column for column in table.schema if column != target_column],
        key="known_columns",
    )

    if not known_columns:
        return

    default_problem = (
        Problem.classification
        if Attribute.categorical in table.schema[target_column]
        else Problem.regression
    )

    problem = streamlit.selectbox(
        "Problem",
        list(Problem),
        index=default_problem.value - 1,
        key="problem",
        on_change=lambda: _clear_state("solver"),
    )

    if not problem:
        return

    solver = streamlit.selectbox(
        "Solver",
        [solver for solver in SOLVERS if solver.problem == problem],
        key="solver",
    )

    if not solver:
        return

    # todo: make this configurable
    transform = (
        Drop(columns={table.name: {target_column}})
        .then(Cast())
        .then(Encode())
        .then(Identity(), Aggregate())
    )

    _apply_transform(dataset, transform)


@streamlit.cache_resource(
    hash_funcs={Dataset: id},
    max_entries=1,
)
def _apply_transform(
    dataset: Dataset,
    transform: Transform,
) -> Dataset:
    return dataset.apply(transform)


def _clear_state(
    *keys: str,
) -> None:
    for key in keys:
        if key in streamlit.session_state:
            del streamlit.session_state[key]
