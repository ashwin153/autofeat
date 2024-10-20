from typing import Optional

import numpy
import pandas
import streamlit

from autofeat.dataset import Dataset
from autofeat.model import Model
from autofeat.table import Table


@streamlit.fragment
def explore_predictions(
    model: Model,
    prediction_dataset: Dataset,
) -> None:
    if known_column_table := _get_known_columns(prediction_dataset, model.known.columns):
        # TODO: Right now there are no features for the new data I am giving the system
        # in the trained model. we need to make sure those IDs somehow have feature values
        prediction = model.predict(known_column_table)
    else:
        return

    results = prediction.X.to_pandas()

    prediction_column = f"{prediction.model.y.name} prediction"
    results[prediction_column] = prediction.y.to_pandas()

    results = results.join(prediction.known.to_pandas())

    # Move prediction column to the front
    cols = [prediction_column] + [col for col in results.columns if col != prediction_column]
    results = results[cols]

    col1, col2 = streamlit.columns(2)

    with col1:
        event = streamlit.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
            column_config={
                "original_index": None,
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    with col2:
        selected_rows = event.get("selection", {}).get("rows", [])
        if selected_rows:
            selected_row = selected_rows[0]
            shap_df = _grid(
                            selected_row,
                            numpy.abs(prediction.explanation.values),
                            prediction.y.to_numpy(),
                            prediction.X.to_pandas(),
                        )

            streamlit.dataframe(
                shap_df,
                column_config={
                    "Source": None,
                    "Importance": streamlit.column_config.ProgressColumn(
                        format="%.2f",
                        max_value=1,
                        min_value=0,
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            streamlit.write("Select a row from the predictions table to see SHAP values.")


# given a Model, a predicted row, and SHAP explanation.values,
# return the SHAP explanation values for each prediciton.
# For multi-class, only return the SHAP value for the predicted class by the model
# Return it sorted by most important feature to least important.
@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=2,
)
def _grid(
    row_index: int,
    shap_values: numpy.ndarray,
    predictions: numpy.ndarray,
    features: pandas.DataFrame,
) -> pandas.DataFrame:
    # Get the SHAP values for the specific row
    row_shap_values = shap_values[row_index]
    # For multi-output models, we need to select the predicted class
    if row_shap_values.ndim == 2:
        predicted_class_index = predictions[row_index].argmax()
        shap_values_selected = row_shap_values[predicted_class_index, :]
    else:
        shap_values_selected = row_shap_values

    # Normalize the SHAP values
    importance = numpy.abs(shap_values_selected)
    importance = importance / numpy.max(importance)

    print(features)
    print(importance)
    # Create a DataFrame with feature names and their SHAP values
    df = pandas.DataFrame({
        "Predictor": [c.split(" :: ", 1)[0] for c in features.columns],
        "Importance": importance,
        "Source": [c.split(" :: ", 1)[1] if " :: " in c else "" for c in features.columns],
        "Feature Value": features.iloc[row_index].values,
    })

    # Sort by importance
    df = df.sort_values("Importance", ascending=False)

    return df


#Find the first set of known columns in a table in the data.
# Currently assumes that one table is being inputted
# TODO: Make work with known columns across tables, etc.
def _get_known_columns(
    prediction_dataset: Dataset,
    known_column_names: list[str],
) -> Optional[Table]:
    for table in prediction_dataset.tables:
        try:
            # Attempt to get all known columns from the current table
            found_columns = [table.column(name) for name in known_column_names]
            return table.select(found_columns)
        except Exception as e:
            # If any column is not found, move to the next table
            print(f"Not all columns found in table {table.name}. Error: {e}")
            continue
    # If we've checked all tables and haven't returned, no matching table was found
    print("No table found with all known columns")
    return None

