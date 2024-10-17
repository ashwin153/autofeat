import streamlit

from autofeat.attribute import Attribute
from autofeat.dataset import Dataset
from autofeat.model import Model
from autofeat.predictor import PREDICTION_METHODS, PredictionMethod
from autofeat.problem import Problem
from autofeat.table import Column, Table
from autofeat.ui.show_log import show_log


def train_model(
    dataset: Dataset,
) -> Model | None:
    """Load features from the ``dataset`` that are relevant to a prediction problem.

    :param dataset: Dataset to load features from.
    """
    training_data = streamlit.selectbox(
        help="Table that contains your training data",
        index=None,
        key="training_data",
        label="Training Data",
        on_change=lambda: _clear_state("target_column"),
        options=dataset.tables,
    )

    if not training_data:
        return None

    target_column = streamlit.selectbox(
        help="Column that you are trying to predict",
        index=None,
        key="target_column",
        label="Target Column",
        on_change=lambda: _clear_state("known_columns", "problem"),
        options=[c for c in training_data.columns if Attribute.not_null in c.attributes],
    )

    if not target_column:
        return None

    known_columns = streamlit.multiselect(
        default=[c for c in training_data.columns if Attribute.primary_key in c.attributes],
        help="Columns that are known at the time of prediction",
        key="known_columns",
        label="Known Columns",
        options=[c for c in training_data.columns if c.name != target_column],
    )

    if not known_columns:
        return None

    default_problem = (
        Problem.classification
        if Attribute.categorical in target_column.attributes
        else Problem.regression
    )

    problem = streamlit.selectbox(
        help="Whether the target variable is categorical (classification) or numeric (regression)",
        index=default_problem.value - 1,
        key="problem",
        label="Problem",
        on_change=lambda: _clear_state("prediction_method"),
        options=list(Problem),
    )

    with streamlit.expander("Configure Methodology"):
        prediction_method = streamlit.selectbox(
            help="Method of predicting the target variable given the input features",
            key="prediction_method",
            label="Prediction Method",
            options=PREDICTION_METHODS,
        )

    with show_log("Training Model"):
        return _train_model(
            dataset=dataset,
            known_columns=tuple(known_columns),
            prediction_method=PREDICTION_METHODS[prediction_method],
            problem=problem,
            training_data=training_data,
            target_column=target_column,
        )


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        Table: lambda x: x.name,
        Column: lambda x: x.name,
    },
    max_entries=1,
    show_spinner=False,
)
def _train_model(
    *,
    dataset: Dataset,
    known_columns: tuple[Column, ...],
    prediction_method: PredictionMethod,
    problem: Problem,
    target_column: Column,
    training_data: Table,
) -> Model:
    return Model.train(
        dataset=dataset,
        known_columns=known_columns,
        prediction_method=prediction_method,
        problem=problem,
        target_column=target_column,
        training_data=training_data,
    )


def _clear_state(
    *keys: str,
) -> None:
    for key in keys:
        if key in streamlit.session_state:
            del streamlit.session_state[key]
