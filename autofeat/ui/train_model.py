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
from autofeat.table import Column, Table
from autofeat.transform import Aggregate, Cache, Combine, Drop, Identity


def train_model(
    dataset: Dataset,
) -> TrainedModel | None:
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
        PredictionProblem.classification
        if Attribute.categorical in target_column.attributes
        else PredictionProblem.regression
    )

    problem = streamlit.selectbox(
        help="Whether the target variable is categorical (classification) or numeric (regression)",
        index=default_problem.value - 1,
        key="problem",
        label="Problem",
        on_change=lambda: _clear_state("prediction_method"),
        options=list(PredictionProblem),
    )

    with streamlit.expander("Configure Methodology"):
        prediction_method = streamlit.selectbox(
            help="Method of predicting the target variable given the input features",
            key="prediction_method",
            label="Prediction Method",
            options=[method for method in PREDICTION_METHODS.values() if method.problem == problem],
        )

        selection_method = streamlit.selectbox(
            help="Method of selecting the most important features to the prediction model",
            key="selection_method",
            label="Selection Method",
            options=SELECTION_METHODS.values(),
        )

    if not streamlit.button("Train Model"):
        return None

    return _train_model(
        dataset=dataset,
        known_columns=tuple(known_columns),
        prediction_method=prediction_method,
        selection_method=selection_method,
        training_data=training_data,
        target_column=target_column,
    )


@streamlit.cache_resource(
    hash_funcs={
        Dataset: id,
        PredictionMethod: lambda x: x.name,
        SelectionMethod: lambda x: x.name,
        Table: id,
        Column: id,
    },
    max_entries=1,
)
def _train_model(
    *,
    dataset: Dataset,
    known_columns: tuple[Column, ...],
    prediction_method: PredictionMethod,
    selection_method: SelectionMethod,
    training_data: Table,
    target_column: Column,
) -> TrainedModel:
    # extract the known and target variables from the training data
    known = (
        training_data.data
        .select([column.name for column in known_columns])
        .collect()
    )

    target = (
        training_data.data
        .select(target_column.name)
        .collect()
    )

    # drop all columns that are related to the target column
    dataset = dataset.apply(Drop(columns=[target_column]))

    # TODO: make this configurable
    # transforms to iteratively apply to the dataset
    rounds = [
        (Identity(), 60),
        (Aggregate(is_pivotable=known_columns), 40),
        (Combine(), 10),
    ]

    cache = Cache()

    i = 0
    while True:
        # apply the next transform to the dataset, train a model, and select the top n features
        transform, n_features = rounds[i]
        dataset = dataset.apply(Identity().then(Identity(), transform).then(cache))
        model = dataset.train(known, target, prediction_method, selection_method, n_features)
        dataset = model.dataset
        cache.keep(dataset)

        if i == len(rounds) - 1:
            return model
        else:
            i += 1


        selection_by_table = collections.defaultdict(set)
        for selected in selection:
            column, table = selected.split(Extract.SEPARATOR, 1)
            selection_by_table[table].add(column)
        dataset = self.apply(Keep(columns=selection_by_table))



def _clear_state(
    *keys: str,
) -> None:
    for key in keys:
        if key in streamlit.session_state:
            del streamlit.session_state[key]
