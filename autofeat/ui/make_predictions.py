import pandas
import polars
import streamlit

from autofeat.model import Model, Prediction


def make_predictions(
    model: Model,
) -> None:
    """Make bulk predictions.

    :param model: Prediction model.
    """
    left,  right = streamlit.columns(2)

    with left:
        csv_file = streamlit.file_uploader(
            label="Known Columns",
            type="CSV",
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        default_known = (
            polars.read_csv(csv_file, columns=model.known.columns)
            if csv_file
            else model.known.head(1)
        )

        known = streamlit.data_editor(
            default_known.to_pandas(),
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
        )

    with right:
        prediction = _make_prediction(model, known)

        streamlit.dataframe(
            polars.concat(
                [prediction.y.to_frame(), prediction.known, prediction.X],
                how="horizontal",
            ),
            height=551,
            use_container_width=True,
        )


@streamlit.cache_resource(
    hash_funcs={
        Model: id,
    },
    max_entries=1,
)
def _make_prediction(
    model: Model,
    known: pandas.DataFrame,
) -> Prediction:
    return model.predict(polars.from_pandas(known, schema_overrides=model.known.schema))
