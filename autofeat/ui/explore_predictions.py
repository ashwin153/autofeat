import numpy
import pandas
import streamlit

from autofeat.model import Model
from autofeat.problem import Problem


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
            selected_row_index = selected_rows[0]
            print(f"{results.iloc[selected_row_index]['LotArea :: train.csv']} prediction for the Lot Area of the selected row")
            shap_df = _grid(model, selected_row_index, shap_values, predictions, results)

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
    features: pandas.DataFrame,
) -> pandas.DataFrame:
    # Get the SHAP values for the specific row
    row_shap_values = shap_values[row_index]
    # For multi-output models, we need to select the predicted class
    predicted_value = 0
    if row_shap_values.ndim == 2:
        predicted_class_index = predictions[row_index].argmax()
        shap_values_selected = row_shap_values[predicted_class_index, :]
        predicted_value = predicted_class_index
    else:
        shap_values_selected = row_shap_values
        predicted_value = predictions[row_index]

      # Normalize the SHAP values
    importance = numpy.abs(shap_values_selected)
    importance = importance / numpy.max(importance)

    # Create a DataFrame with feature names and their SHAP values
    df = pandas.DataFrame({
        "Predictor": model.X.columns,
        "Importance": importance,
        "Source": [c.split(" :: ", 1)[1] if " :: " in c else "" for c in model.X.columns],
    })

    # Analyze numerical features
    for feature in df["Predictor"]:
        if model.X.schema[feature].is_numeric():
            print(f"feature value: {features[feature].iloc[row_index]}")
            best_change, best_bounds = max([bucket_stats(feature, n, model, row_index, predicted_value, features) for n in [4]], key=lambda x: abs(x[0]))
            df.loc[df["Predictor"] == feature, "Bucket Change"] = best_change
            df.loc[df["Predictor"] == feature, "Bucket Bounds"] = str(best_bounds)

    # Sort by importance
    df = df.sort_values("Importance", ascending=False)

    return df


def bucket_stats(feature, n_splits, model, row_index, predicted_value, features):
    # Extract the feature column from the features DataFrame
    feature_values = pandas.to_numeric(features[feature], errors='coerce').to_numpy()
    y_values = model.y.to_numpy()

    # Remove NaN values
    mask = ~numpy.isnan(feature_values)
    feature_values = feature_values[mask]
    y_values = y_values[mask]

    if len(feature_values) == 0:
        print(f"No valid numeric values for feature {feature}")
        return 0, (None, None)

    # Get the original feature value for the specific row
    original_feature_value = pandas.to_numeric(features[feature].iloc[row_index], errors='coerce')

    splits = numpy.percentile(feature_values, numpy.linspace(0, 100, n_splits+1))

    # Use the original feature value to determine the bucket
    bucket = numpy.digitize([original_feature_value], splits) - 1
    if bucket == n_splits:
        bucket = n_splits - 1

    bucket_mask = (feature_values > splits[bucket]) & (feature_values <= splits[bucket+1])

    if model.problem == Problem.regression:
        overall_stat = numpy.nanmean(y_values)
        bucket_stat = numpy.nanmean(y_values[bucket_mask])
    else:  # Classification (binary or multiclass)
        overall_stat = numpy.nanmean(y_values == predicted_value)
        bucket_stat = numpy.nanmean(y_values[bucket_mask] == predicted_value)

    change = (bucket_stat - overall_stat) / overall_stat * 100 if overall_stat != 0 else 0

    # Debug information
    print(f"Feature: {feature}")
    print(f"Feature value: {original_feature_value}")
    print(f"Bucket: {bucket}")
    print(f"Bucket bounds: ({splits[bucket]}, {splits[bucket+1]})")
    print(f"Number of values in bucket: {numpy.sum(bucket_mask)}")
    print(f"Overall stat: {overall_stat}")
    print(f"Bucket stat: {bucket_stat}")
    print(f"Change: {change}")
    print("Splits:", splits)
    print("---")

    return change, (splits[bucket], splits[bucket+1])
