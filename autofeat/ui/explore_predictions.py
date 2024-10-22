from typing import Any

import numpy
import pandas
import polars
import streamlit
import streamlit.elements.lib.column_types

from autofeat.model import Model, Prediction
from autofeat.transform import Extract


@streamlit.fragment
def explore_predictions(
    model: Model,
) -> None:
    left, middle, right = streamlit.columns(3)

    with left:
        known = _into_input_widgets(model.known)

    with middle:
        _into_input_widgets(_extract_features(model, known))

    with right:
        # TODO: this should use `features` instead of `known`
        prediction = _make_prediction(model, known)

        streamlit.metric(
            label=prediction.y.name,
            value=prediction.y[0],
        )

        streamlit.dataframe(
            _explain_prediction(prediction),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Importance": streamlit.column_config.ProgressColumn(
                    format="%.2f",
                    max_value=1,
                    min_value=0,
                ),
            },
        )


def _into_input_widgets(
    df: polars.DataFrame,
) -> polars.DataFrame:
    return polars.DataFrame(
        [
            {
                column: _into_input_widget(column, data_type, default)
                # TODO: they should be sorted in order of importance
                for (column, data_type), default in sorted(zip(df.schema.items(), df.row(0)))
            },
        ], schema=df.schema,
    )


def _into_input_widget(
    column: str,
    data_type: polars.DataType,
    default: Any,
) -> Any:
    parts = column.split(Extract.SEPARATOR, 1)
    label = parts[0]
    help = parts[1] if len(parts) > 1 else None

    if isinstance(data_type, polars.Boolean):
        return streamlit.checkbox(
            label=label,
            help=help,
            value=default,
        )
    elif data_type.is_temporal():
        return streamlit.date_input(
            label=label,
            help=help,
            value=default,
        )
    elif data_type.is_numeric():
        return streamlit.number_input(
            label=label,
            help=help,
            value=default,
        )
    else:
        return streamlit.text_input(
            label=label,
            help=help,
            value=default,
        )


@streamlit.cache_data(
    hash_funcs={
        Model: id,
        polars.DataFrame: polars.DataFrame.to_pandas,
    },
    max_entries=1,
)
def _extract_features(
    model: Model,
    known: polars.DataFrame,
) -> polars.DataFrame:
    return model.dataset.features(known)


@streamlit.cache_resource(
    hash_funcs={
        Model: id,
        polars.DataFrame: polars.DataFrame.to_pandas,
    },
    max_entries=1,
)
def _make_prediction(
    model: Model,
    known: polars.DataFrame,
) -> Prediction:
    return model.predict(known)


@streamlit.cache_data(
    hash_funcs={Prediction: id},
    max_entries=1,
)
def _explain_prediction(
    prediction: Prediction,
) -> pandas.DataFrame:
    importance = (
        numpy.abs(prediction.explanation.values).mean((0, 2))
        if len(prediction.explanation.shape) == 3
        else numpy.abs(prediction.explanation.values).mean(0)
    )

    importance = importance / numpy.max(importance)

    feature_names = [
        feature_name.split(Extract.SEPARATOR, 1)
        for feature_name in prediction.explanation.feature_names
    ]

    df = pandas.DataFrame({
        "Feature": [column_name for column_name, _ in feature_names],
        "Importance": importance,
    })

    df = df.sort_values("Importance", ascending=False)

    return df
