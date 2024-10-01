import streamlit

from autofeat.attribute import Attribute
from autofeat.dataset import Dataset
from autofeat.model import (
    PREDICTION_METHODS,
    SELECTION_METHODS,
    PredictionMethod,
    PredictionProblem,
    SelectionMethod,
    TrainedModel,
)
from autofeat.table import Table
from autofeat.transform import Aggregate, Cast, Drop, Encode, Identity, Transform


def train_model(
    dataset: Dataset,
) -> TrainedModel | None:
    """Load features from the ``dataset`` that are relevant to a prediction problem.

    :param dataset: Dataset to load features from.
    """
    streamlit.header("Train Model")

    table = streamlit.selectbox(
        "Table",
        dataset.tables,
        index=None,
        key="table",
        on_change=lambda: _clear_state("target_column"),
    )

    if not table:
        return None

    target_column = streamlit.selectbox(
        "Target Column",
        table.schema,
        index=None,
        key="target_column",
        on_change=lambda: _clear_state("known_columns", "problem"),
    )

    if not target_column:
        return None

    known_columns = streamlit.multiselect(
        "Known Columns",
        [column for column in table.schema if column != target_column],
        key="known_columns",
    )

    if not known_columns:
        return None

    default_problem = (
        PredictionProblem.classification
        if Attribute.categorical in table.schema[target_column]
        else PredictionProblem.regression
    )

    problem = streamlit.selectbox(
        "Problem",
        list(PredictionProblem),
        index=default_problem.value - 1,
        key="problem",
        on_change=lambda: _clear_state("prediction_method"),
    )

    prediction_method = streamlit.selectbox(
        "Prediction Method",
        [method for method in PREDICTION_METHODS.values() if method.problem == problem],
        key="prediction_method",
    )

    selection_method = streamlit.selectbox(
        "Selection Method",
        SELECTION_METHODS.values(),
        key="selection_method",
    )

    # todo: make this configurable
    transform = (
        Drop(columns={table.name: {target_column}})
        .then(Cast())
        .then(Encode())
        .then(Identity(), Aggregate())
    )

    if not streamlit.button("Train Model"):
        return None

    return _train_model(
        dataset=dataset,
        known_columns=tuple(known_columns),
        prediction_method=prediction_method,
        selection_method=selection_method,
        table=table,
        target_column=target_column,
        transform=transform,
    )


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        PredictionMethod: lambda x: x.name,
        SelectionMethod: lambda x: x.name,
        Table: id,
    },
    max_entries=1,
)
def _train_model(
    *,
    dataset: Dataset,
    known_columns: tuple[str, ...],
    prediction_method: PredictionMethod,
    selection_method: SelectionMethod,
    table: Table,
    target_column: str,
    transform: Transform,
) -> TrainedModel:
    inputs = dataset.apply(transform)

    return inputs.train(
        known=table.data.select(known_columns),
        target=table.data.select(target_column),
        prediction_method=prediction_method,
        selection_method=selection_method,
    )


def _clear_state(
    *keys: str,
) -> None:
    for key in keys:
        if key in streamlit.session_state:
            del streamlit.session_state[key]
