import math
from typing import Any

import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
import shap
import sklearn.metrics
import streamlit

from autofeat.model import PredictionProblem, TrainedModel


def evaluate_model(
    model: TrainedModel,
) -> None:
    match model.prediction_method.problem:
        case PredictionProblem.classification:
            metrics = _classification_metrics(model.y_test, model.y_predicted)
            baseline = _classification_metrics(model.y_test, model.y_baseline)
            improvement = _percent_change(baseline["accuracy"], metrics["accuracy"])

            headline = f"✅ Model is **:green[{improvement:.2f}% more accurate]** than always guessing the most frequent category" # noqa: E501
            table_data = {
                "Metric": ["Accuracy", "Precision", "Recall"],
                "Model": [
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                ],
                "Baseline": [
                    f"{baseline['accuracy']:.4f}",
                    f"{baseline['precision']:.4f}",
                    f"{baseline['recall']:.4f}",
                ],
                "Improvement (%)": [
                    f"{_percent_change(baseline['accuracy'], metrics['accuracy']):.2f}%",
                    f"{_percent_change(baseline['precision'], metrics['precision']):.2f}%",
                    f"{_percent_change(baseline['recall'], metrics['recall']):.2f}%",
                ],
            }
        case PredictionProblem.regression:
            metrics = _regression_metrics(model.y_test, model.y_predicted)
            baseline = _regression_metrics(model.y_test, model.y_baseline)
            improvement = -_percent_change(baseline["rmse"], metrics["rmse"])

            headline = f"✅ Model is **:green[{improvement:.2f}% more accurate]** than always guessing the mean" # noqa: E501
            table_data = {
                "Metric": ["RMSE", "R2"],
                "Model": [
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['r2']:.4f}",
                ],
                "Baseline": [
                    f"{baseline['rmse']:.4f}",
                    f"{baseline['r2']:.4f}",
                ],
                "Improvement (%)": [
                    f"{_percent_change(baseline['rmse'], metrics['rmse']):.2f}%",
                    f"{_percent_change(baseline['r2'],metrics['r2']):.2f}%",
                ],
            }
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

    # Streamlit bordered section with title and headline
    with streamlit.container(border=True):
        streamlit.subheader("Model Performance")
        streamlit.markdown(f"{headline}")

        # Expander with a table of model stats
        with streamlit.expander("Show detailed model stats"):
            streamlit.dataframe(table_data, hide_index=True, use_container_width=True)

            # Light text about interpreting the metrics
            match model.prediction_method.problem:
                case PredictionProblem.classification:
                    streamlit.caption(
                        f"Comparison: model's performance against a baseline model that randomly guesses based on the frequency of {model.y.name} values. " # noqa: E501
                        "Higher precision indicates a model guesses the value correctly more on average. " # noqa: E501
                        "Higher recall indicates that a model covers more correct classifications overall."  # noqa: E501
                    )
                case PredictionProblem.regression:
                    streamlit.caption(
                        f"Comparison: the model's performance against a baseline model, that always predicts the mean of {model.y.name}. "  # noqa: E501
                        "RMSE is the average error between the actual value and the prediction by the model. Lower error is better (means guesses are closer to true). " # noqa: E501
                        "R2 indicates how much of the variation in your data the model captures. A higher value is better."  # noqa: E501
                    )

    _create_feature_charts(model)


def _create_feature_charts(
    model: TrainedModel,
) -> None:
    _create_feature_importance_charts(model)
    _create_feature_analysis_charts(model)


@streamlit.fragment
def _create_feature_importance_charts(
    model: TrainedModel,
) -> None:
    # Generate feature importances and sort them in descending order
    feature_importance = _feature_importance(model)

    # Extract 'Column' and 'Table' information, then rename columns
    feature_importance["Predictor"] = [c.split(" :: ", 1)[0] for c in model.X.columns]
    feature_importance["Table Sources"] = [c.split(" :: ", 1)[1] for c in model.X.columns]

    # Normalize the importance values
    max_importance = feature_importance["Importance"].max()
    feature_importance["Importance"] = feature_importance["Importance"] / max_importance

    # Sort by importance in descending order
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    # Reorder columns
    feature_importance = feature_importance[["Predictor", "Importance", "Table Sources"]]

    with streamlit.container(border=True):
        streamlit.subheader(f"Predictors of {model.y.name}")
        streamlit.caption(
            f"This chart displays the top predictors of {model.y.name}. "
            "The importance of each predictor indicates how predictive it is relative to the others.", # noqa: E501
        )
        # Display the DataFrame in Streamlit with a progress column for Importance
        streamlit.dataframe(
            feature_importance,
            column_config={
                "Importance": streamlit.column_config.ProgressColumn(
                    format="%.2f",
                    max_value=1,
                    min_value=0,
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

@streamlit.fragment
def _create_feature_analysis_charts(
    model: TrainedModel,
) -> None:
    feature_importance = _feature_importance(model)
    feature_importance = feature_importance.sort_values("Importance", ascending=False)
    feature_list = feature_importance["Feature"].tolist()

    # Initialize session state for tabs
    if "tabs" not in streamlit.session_state:
        streamlit.session_state.tabs = []
    else:
        streamlit.session_state.tabs = [tab for tab in streamlit.session_state.tabs if tab["feature"] in feature_list] #noqa 

    tabs_list = streamlit.session_state.tabs
    tab_labels = ["Create Chart"] + [tab["label"] for tab in tabs_list]  # 'New Tab' is now first

    # Use streamlit.tabs to create tabbed interface
    tabs = streamlit.tabs(tab_labels)

    # Handle the 'New Tab' for adding new tabs
    with tabs[0]:  # Index 0 for 'New Tab'
        available_features = [
            f for f in feature_list
            if f not in [t.get("feature") for t in streamlit.session_state.tabs if t.get("feature")]
        ]
        if available_features:
            streamlit.subheader("Create a new chart")
            selected_feature = streamlit.selectbox(
                "Select a predictor to analyze:",
                available_features,
                index=None,
            )
            if selected_feature is not None:
                # Add a new tab with the selected feature
                new_tab = {"label": selected_feature, "feature": selected_feature}
                streamlit.session_state.tabs.append(new_tab)
                streamlit.rerun(scope="fragment")
        else:
            streamlit.write("No more features to select.")

    # Handle existing tabs
    for idx, tab in enumerate(tabs_list):
        with tabs[idx + 1]:  # Existing tabs start from index 1
            selected_feature = tab["feature"]

            # Generate the chart based on the prediction problem
            if model.prediction_method.problem == PredictionProblem.classification:
                streamlit.subheader(f"{selected_feature} vs. {model.y.name}")
                chart_fig = _create_classification_feature_chart(model, selected_feature)
            elif model.prediction_method.problem == PredictionProblem.regression:
                streamlit.subheader(f"{selected_feature} vs. {model.y.name}")
                chart_fig = _create_regression_feature_chart(model, selected_feature)
            else:
                raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

            streamlit.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})  # noqa: E501

            # Button to close the tab
            if streamlit.button("Close Tab", key=f"close_{idx}"):
                del streamlit.session_state.tabs[idx]
                streamlit.rerun(scope="fragment")


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def feature_selection_form(
    model: TrainedModel,
) -> None:
    return


def _create_classification_feature_chart(  # type: ignore[no-any-unimported]
    model: TrainedModel,
    feature: str,
) -> go.Figure:
    # Get the index of the feature
    i = model.X.columns.index(feature)
    # Clean the data
    x, y_true = _clean_data(model.X_test[:, i], model.y_test)
    # Check if we have any data left after cleaning
    if len(x) == 0:
        return go.Figure()

    fig = go.Figure()
    df = pandas.DataFrame({"feature": x, f"{model.y.name}": y_true})
    # Check if the feature is numerical
    if model.X.schema[feature].is_numeric():
        # Numerical feature: create a histogram with adjustable buckets
        num_buckets = 4
        # Sort the data and create buckets
        sorted_data = df.sort_values("feature")
        bucket_size = len(sorted_data) // num_buckets

        bucket_data = []
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i < num_buckets - 1 else len(sorted_data)
            bucket = sorted_data.iloc[start_idx:end_idx]

            bucket_name = f"{bucket['feature'].min():.2f} - {bucket['feature'].max():.2f}"
            for target_value in df[model.y.name].unique():
                count = bucket[bucket[model.y.name] == target_value].shape[0]
                bucket_data.append({
                    "Bucket": bucket_name,
                    model.y.name: target_value,
                    "Count": count,
                    "BucketStart": bucket['feature'].min(),
                    "BucketEnd": bucket['feature'].max(),
                })

        bucket_df = pandas.DataFrame(bucket_data)

        fig = px.bar(bucket_df, x="Bucket", y="Count", color=model.y.name,
                     labels={"Bucket": feature, "Count": f"{target_value} count"},
                     height=600, width=800)

        fig.update_layout(bargap=0.2)
        fig.update_xaxes(title_text=f"{feature} (Buckets)")
        fig.update_yaxes(title_text=f"{model.y.name} count")
    else:
        # Categorical feature: create a normalized stacked bar chart
        df_counts = df.groupby(["feature", f"{model.y.name}"]).size().unstack(fill_value=0)
        df_percentages = df_counts.apply(lambda x: x / x.sum() * 100, axis=1)

        fig = px.bar(
            df_percentages,
            x=df_percentages.index,
            y=df_percentages.columns,
            barmode="stack",
            labels={"x": feature, "y": "Percentage", "color": f"{model.y.name}"},
        )

        fig.update_layout(
            xaxis_title=feature,
            yaxis_title=f"{model.y.name} (%)",
            height=600,
            width=800,
            yaxis={"tickformat": ".1f", "range": [0, 100]},
            xaxis={"type": "category", "categoryorder": "total descending"},
            margin=dict(t=20),
        )

    return fig


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _create_regression_feature_chart(  # type: ignore[no-any-unimported]
    model: TrainedModel,
    feature: str,
) -> go.Figure:
    # Get the index of the feature
    i = model.X.columns.index(feature)
    # Clean the data
    x, y_true = _clean_data(model.X_test[:, i], model.y_test)
    # Check if we have any data left after cleaning
    if len(x) == 0:
        return go.Figure()

    # Create the Plotly figure
    fig = go.Figure()

    # Check if the feature is numerical
    if model.X.schema[feature].is_numeric():
        # Numerical feature: create a scatter plot
        x = pandas.to_numeric(x, errors="coerce")
        mask = ~numpy.isnan(x)
        x = x[mask]
        y_true = y_true[mask]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_true,
                mode="markers",
                name=f"{model.y.name}",
                marker={
                    "size": 5,
                    "color": "blue",
                    "opacity": 0.6,
                },
            ),
        )
        # Add line of best fit
        coeffs = numpy.polyfit(x, y_true, 1)
        line_x = numpy.array([numpy.min(x), numpy.max(x)])
        line_y = coeffs[0] * line_x + coeffs[1]

        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Line of Best Fit",
                line={"color": "red", "width": 2},
            ),
        )

    else:
        # Non-numerical feature: create a box plot
        fig.add_trace(
            go.Box(
                x=x,
                y=y_true,
                name="Distribution",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ),
        )

    # Customize the layout
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title=f"{model.y.name}",
        height=400,
        width=600,
        margin=dict(t=30),
    )

    return fig

@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _classification_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn.metrics.precision_score(y_true, y_pred, average="weighted"),
        "recall": sklearn.metrics.recall_score(y_true, y_pred, average="weighted"),
    }


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _regression_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
) -> dict[str, Any]:
    return {
        "r2": sklearn.metrics.r2_score(y_true, y_pred),
        "rmse": math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)),
    }


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _shap_explanation(  # type: ignore[no-any-unimported]
    model: TrainedModel,
) -> shap.Explanation:
    shap_explainer = shap.Explainer(
        model.prediction_model,
        feature_names=model.X.columns,
    )

    return shap_explainer(model.X_test)


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _feature_importance(
    model: TrainedModel,
) -> pandas.DataFrame:
    shap_explanation = _shap_explanation(model)

    importance = (
        numpy.abs(shap_explanation.values).mean((0, 2))
        if len(shap_explanation.shape) == 3
        else numpy.abs(shap_explanation.values).mean(0)
    )

    return pandas.DataFrame(
        {
            "Feature": model.X.columns,
            "Importance": importance,
        },
    )


def _percent_change(
    old: float,
    new: float,
) -> float:
    return (new - old) / old * 100



def _clean_data(
    x: Any,
    y: Any,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    # Convert to numpy arrays if they aren't already
    x = numpy.array(x)
    y = numpy.array(y)

    # Create a mask for non-null and non-NaN values
    mask = ~(pandas.isnull(x) | pandas.isnull(y))

    return x[mask], y[mask]
