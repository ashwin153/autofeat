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

            headline = f"✅ Model is {improvement:.2f}% more accurate than always guessing randomly"
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

            headline = f"✅ Model is {improvement:.2f}% more accurate than always guessing the mean"
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
                    f"{_percent_change(baseline['r2'], metrics['r2']):.2f}%",
                ],
            }
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

    # Streamlit bordered section with title and headline
    with streamlit.container(border=True):
        streamlit.subheader("Model Performance")
        streamlit.markdown(f"**{headline}**")

        # Expander with a table of model stats
        with streamlit.expander("Show detailed model stats"):
            streamlit.table(table_data)

            # Light text about interpreting the metrics
            streamlit.caption(
                "The metrics shown compare the model's performance against a baseline model. "
                "Improvements are calculated as percentage changes from the baseline. "
                "Higher accuracy, precision, recall, or R2, and lower RMSE, indicate better performance.", # noqa: E501
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
    #generate feature importances and sort them in descending order
    feature_importance = _feature_importance(model)
    feature_importance = feature_importance.sort_values("Importance", ascending=False)
    # Get the maximum importance for consistent x-axis range
    max_importance = feature_importance["Importance"].max()

    # Page size (fixed to 8 for this case)
    batch_size = 8

    # Determine total pages
    total_pages = max(1, (len(feature_importance) - 1) // batch_size + 1)
    with streamlit.container(border=True):
        streamlit.subheader(f"Predictors of {model.y.name}")
        streamlit.caption(
            f"This chart displays the top predictors of {model.y.name}. "
            "The importance of each predictor indicates how predictive it is relative to the others.", # noqa: E501
        )
        with streamlit.container():
            # Create the bottom menu for pagination controls
            pagination = streamlit.container()
            bottom_menu = streamlit.columns([3, 1])
            # Page number input with steppers
            with bottom_menu[1]:
                current_page = streamlit.number_input(
                    "Page", min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    label_visibility="collapsed",
                )

            # Display current page info
            with bottom_menu[0]:
                streamlit.markdown(f"Page **{current_page}** of **{total_pages}**")

            # Function to split the dataset into pages
            def split_frame(df, batch_size) -> list: # noqa: no-untyped-def
                return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

            # Split the dataset and get the current page's data
            pages = split_frame(feature_importance, batch_size)
            current_page_data = pages[current_page - 1]

            # Create the Plotly bar chart for the selected subset
            fig = go.Figure(
                go.Bar(
                    x=current_page_data["Importance"],
                    y=current_page_data["Feature"],
                    orientation="h",
                    marker_color="steelblue",
                ),
            )

            # Update layout for better appearance and maintain a consistent x-axis range
            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                height=40 * len(current_page_data),
                yaxis={"autorange": "reversed"},
                xaxis_range=[0, max_importance],  # Set consistent x-axis range
            )

            # Display the chart
            pagination.plotly_chart(fig, use_container_width=True)


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
    if "selected_tab_index" not in streamlit.session_state:
        streamlit.session_state.selected_tab_index = len(streamlit.session_state.tabs) - 1 #first tab default # noqa: E501

    tabs_list = streamlit.session_state.tabs
    tab_labels = ["New Tab"] + [tab["label"] for tab in tabs_list]  # 'New Tab' is now first

    # Use streamlit.tabs to create tabbed interface
    tabs = streamlit.tabs(tab_labels)

    # Handle the 'New Tab' for adding new tabs
    with tabs[0]:  # Index 0 for 'New Tab'
        available_features = [
            f for f in feature_list
            if f not in [t.get("feature") for t in streamlit.session_state.tabs if t.get("feature")]
        ]
        if available_features:
            streamlit.subheader("Add a New Chart")
            selected_feature = streamlit.selectbox(
                "Select a feature to analyze:",
                ["", *available_features],
                index=0,  # Start with the empty string selected
            )
            if selected_feature != "":
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
            streamlit.subheader(f"Feature: {selected_feature}")

            # Generate the chart based on the prediction problem
            if model.prediction_method.problem == PredictionProblem.classification:
                chart_fig = _create_classification_feature_chart(model, selected_feature)
            elif model.prediction_method.problem == PredictionProblem.regression:
                chart_fig = _create_regression_feature_chart(model, selected_feature)
            else:
                raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

            streamlit.plotly_chart(chart_fig, use_container_width=True)

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

@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _create_classification_feature_chart( # noqa: no-any-unimported
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
    df = pandas.DataFrame({"feature": x, "target": y_true})
    # Check if the feature is numerical
    if model.X.schema[feature].is_numeric():
        # Numerical feature: create a histogram
        fig = px.histogram(
            df, x="feature", color="target",
            hover_data=df.columns,
            title=f"{feature} vs {model.y.name}",
            labels={"feature": feature, "target": f"{model.y.name} Class"},
            height=600, width=800,
        )

        fig.update_layout(bargap=0.2)
        fig.update_yaxes(title_text=f"{model.y.name} Frequency")
    else:
        # Categorical feature: create a normalized stacked bar chart
        df_counts = df.groupby(["feature", "target"]).size().unstack(fill_value=0)
        df_percentages = df_counts.apply(lambda x: x / x.sum() * 100, axis=1)

        fig = px.bar(
            df_percentages,
            x=df_percentages.index,
            y=df_percentages.columns,
            barmode="stack",
            labels={"x": feature, "y": "Percentage", "color": f"{model.y.name} Class"},
            title=f"{feature} vs. {model.y.name}",
        )

        fig.update_layout(
            xaxis_title=feature,
            yaxis_title=f"{model.y.name} Distribution (%)",
            height=600,
            width=800,
            yaxis={"tickformat": ".1f", "range": [0, 100]},
            xaxis={"type": "category", "categoryorder": "total descending"},
        )

    return fig


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _create_regression_feature_chart( # noqa: no-any-unimported
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
        title=f"{feature} vs {model.y.name}",
        xaxis_title=feature,
        yaxis_title=f"{model.y.name}",
        height=400,
        width=600,
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
