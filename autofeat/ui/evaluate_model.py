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

            with streamlit.expander(
                f"✅ model is {improvement:.2f}% more accurate than always guessing randomly",
            ):
                column1, column2, column3 = streamlit.columns(3)

                column1.metric(
                    "Accuracy",
                    value=f"{metrics['accuracy']:.4f}",
                    delta=f"{_percent_change(baseline['accuracy'], metrics['accuracy']):.2f}%",
                )

                column2.metric(
                    "Precision",
                    value=f"{metrics['precision']:.4f}",
                    delta=f"{_percent_change(baseline['precision'], metrics['precision']):.2f}%",
                )

                column3.metric(
                    "Recall",
                    value=f"{metrics['recall']:.4f}",
                    delta=f"{_percent_change(baseline['recall'], metrics['recall']):.2f}%",
                )
        case PredictionProblem.regression:
            metrics = _regression_metrics(model.y_test, model.y_predicted)
            baseline = _regression_metrics(model.y_test, model.y_baseline)
            improvement = -_percent_change(baseline["rmse"], metrics["rmse"])

            with streamlit.expander(
                f"✅ model is {improvement:.2f}% more accurate than always guessing the mean",
            ):
                column1, column2 = streamlit.columns(2)

                column1.metric(
                    "RMSE",
                    value=f"{metrics['rmse']:.4f}",
                    delta=f"{_percent_change(baseline['rmse'], metrics['rmse']):.2f}%",
                )

                column2.metric(
                    "R2",
                    value=f"{metrics['r2']:.4f}",
                    delta=f"{_percent_change(baseline['r2'], metrics['r2']):.2f}%",
                )
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

    _create_feature_charts(model)


@streamlit.fragment
def _create_feature_charts(
    model: TrainedModel,
) -> None:

    #generate feature importances and sort them in descending order
    feature_importance = _feature_importance(model)
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    # Create the Plotly bar chart
    fig = go.Figure(
        go.Bar(
            x=feature_importance["Importance"],
            y=feature_importance["Feature"],
            orientation="h",
            marker_color="steelblue",
        ),
    )

    # Update layout for better appearance
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height= max(600, 40*len(feature_importance)),
        yaxis={"autorange": "reversed"},
    )

    streamlit.plotly_chart(fig, use_container_width=True)
    ordered_list = feature_importance["Feature"].tolist()

    with streamlit.form("feature_selection_form"):
        # Create a dropdown for feature selection
        selected_feature = streamlit.selectbox(
            "Select a feature to analyze:",
            ordered_list,
        )

        submit_button = streamlit.form_submit_button("Show Chart")

        fig = go.Figure()

        if submit_button:
            # Create the corresponding chart based on the selected feature
            match model.prediction_method.problem:
                case PredictionProblem.classification:
                    fig = _create_classification_feature_chart(model, selected_feature)
                    streamlit.plotly_chart(fig, use_container_width=True)
                case PredictionProblem.regression:
                    fig = _create_regression_feature_chart(model, selected_feature)
                    streamlit.plotly_chart(fig, use_container_width=True)
                case _:
                    raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


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
    df = pandas.DataFrame({"feature": x, "target": y_true})
    # Check if the feature is numerical
    if model.X.schema[feature].is_numeric():
        # Numerical feature: create a histogram
        fig = px.histogram(
            df, x="feature", color="target",
            hover_data=df.columns,
            title=f"Feature Analysis: {feature} vs Target",
            labels={"feature": feature, "target": "Target Class"},
            height=600, width=800,
        )

        fig.update_layout(bargap=0.2)
    else:
        # Categorical feature: create a normalized stacked bar chart
        df_counts = df.groupby(["feature", "target"]).size().unstack(fill_value=0)
        df_percentages = df_counts.apply(lambda x: x / x.sum() * 100, axis=1)

        fig = px.bar(
            df_percentages, barmode="stack",
            labels={"value": "Percentage", "target": "Target Class"},
            title=f"Feature Analysis: {feature} vs. Target",
        )

        fig.update_layout(
            xaxis_title=feature,
            yaxis_title="Percentage",
            height=600,
            width=800,
            yaxis={"tickformat": ".1f", "range": [0, 100]},  # Ensure y-axis is 0-100%
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
                name="Data Points",
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
        title=f"Feature Analysis: {feature}",
        xaxis_title=feature,
        yaxis_title="Target Variable",
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
