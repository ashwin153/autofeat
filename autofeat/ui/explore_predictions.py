import numpy
import pandas
import streamlit

from autofeat.model import Model


@streamlit.fragment
def explore_predictions(
    model: Model,
    shap_values: numpy.ndarray,
    predictions: numpy.ndarray,
    feature_values: numpy.ndarray,
) -> None:
    # Prepare the results dataframe
    results = pandas.DataFrame(feature_values, columns=model.X.columns)
    prediction_column = f"{model.y.name} prediction"
    results[prediction_column] = predictions

    # Add original_index column to map back to shap_values and predictions
    results["original_index"] = numpy.arange(len(feature_values), dtype=int)

    # Move prediction column to the front
    cols = [prediction_column] + [col for col in results.columns if col != prediction_column]
    results = results[cols]

    # Create two columns
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
            original_index = int(results.iloc[selected_row]["original_index"])
            shap_df = _grid(model, original_index, shap_values, predictions)

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
    max_entries=5,
)
def _grid(
    model: Model,
    row_index: int,
    shap_values: numpy.ndarray,
    predictions: numpy.ndarray,
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

    # Create a DataFrame with feature names and their SHAP values
    df = pandas.DataFrame({
        "Predictor": [c.split(" :: ", 1)[0] for c in model.X.columns],
        "Importance": importance,
        "Source": [c.split(" :: ", 1)[1] if " :: " in c else "" for c in model.X.columns],
    })

    # Sort by importance
    df = df.sort_values("Importance", ascending=False)

    return df
