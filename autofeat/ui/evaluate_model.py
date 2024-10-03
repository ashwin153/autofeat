import math
from collections import Counter
from typing import Any

import numpy
import pandas
import plotly.graph_objects as go
import shap
import sklearn.metrics
import streamlit

from autofeat.model import PredictionProblem, TrainedModel


def evaluate_model(
    model: TrainedModel,
) -> None:
    streamlit.header("Evaluate Model")

    match model.prediction_method.problem:
        case PredictionProblem.classification:
            metrics = _classification_metrics(model.y_test, model.y_predicted)
            baseline = _classification_metrics(model.y_test, model.y_baseline)
            improvement = _percent_change(baseline["accuracy"], metrics["accuracy"])

            with streamlit.expander(
                f"✅ model is {improvement:.2f}% more accurate than always guessing the most frequent category",  # noqa: E501
            ):
                streamlit.metric(
                    "Accuracy",
                    value=f"{metrics['accuracy']:.4f}",
                    delta=f"{_percent_change(baseline['accuracy'], metrics['accuracy']):.2f}%",
                )

                streamlit.metric(
                    "Precision",
                    value=f"{metrics['precision']:.4f}",
                    delta=f"{_percent_change(baseline['precision'], metrics['precision']):.2f}%",
                )

                streamlit.metric(
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
                streamlit.metric(
                    "RMSE",
                    value=f"{metrics['rmse']:.4f}",
                    delta=f"{_percent_change(baseline['rmse'], metrics['rmse']):.2f}%",
                )

                streamlit.metric(
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
    fig = go.Figure(go.Bar(
        x=feature_importance["Importance"],
        y=feature_importance["Feature"],
        orientation="h",
        marker_color="steelblue",
    ))

    # Update layout for better appearance
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=0, r=0, t=30, b=0),
        height= max(600, 40*len(feature_importance)),
        yaxis=dict(autorange="reversed"),
    )

    # Display the chart, with clickable event (which we can use to do dynamic things in the future)
    streamlit.plotly_chart(fig, use_container_width=True)

    for feature in feature_importance["Feature"]:
        with streamlit.expander(f"Feature Analysis: {feature}"):
            match model.prediction_method.problem:
                case PredictionProblem.classification:
                    chart = _create_classification_feature_chart(model, feature)
                    streamlit.plotly_chart(chart, use_container_width=True)
                case PredictionProblem.regression:
                    chart = _create_regression_feature_chart(model, feature)
                    streamlit.plotly_chart(chart, use_container_width=True)




@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _create_classification_feature_chart(
    model: TrainedModel,
    feature: str,
) -> go.Figure:
    # Get the index of the feature
    i = model.X.columns.index(feature)

    # Get the feature values from X_test
    x = model.X_test[:, i]

    # Get the corresponding actual y values (class labels)
    y_true = model.y_test

    fig = go.Figure()

    # Check if the feature is numerical
    if numpy.issubdtype(x.dtype, numpy.number):
        for cls in numpy.unique(y_true):
            fig.add_trace(go.Box(
                y=x[y_true == cls],
                name=f"Class {cls}",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ))

        fig.update_layout(
            title=f"Feature Analysis: {feature} vs Target",
            yaxis_title=feature,
            xaxis_title="Class",
            height=600,
            width=800,
        )
    else:
        # Calculate percentages
        percentages = {}
        classes = numpy.unique(y_true)
        categories = numpy.unique(x)

        for cls in classes:
            class_data = x[y_true == cls]
            counts = Counter(class_data)
            total = len(class_data)
            percentages[cls] = {cat: counts[cat] / total * 100 for cat in categories}

        for cls in percentages:
            fig.add_trace(go.Bar(
                x=list(categories),
                y=[percentages[cls][cat] for cat in categories],
                name=f"Class {cls}"
            ))

        fig.update_layout(
            title=f"Feature Analysis: {feature} (Categorical)",
            xaxis_title=feature,
            yaxis_title="Percentage",
            barmode="stack",
            height=600,
            width=800,
            yaxis=dict(tickformat=".0%")
        )

    return fig


@streamlit.cache_data(
    hash_funcs={TrainedModel: id},
    max_entries=1,
)
def _create_regression_feature_chart(
    model: TrainedModel,
    feature: str,
) -> go.Figure:
    # Get the index of the feature
    i = model.X.columns.index(feature)

    # Get the feature values from X_test
    x = model.X_test[:, i]

    # Get the corresponding actual y values
    y_true = model.y_test

    # Create the Plotly figure
    fig = go.Figure()

    # Check if the feature is numerical
    if numpy.issubdtype(x.dtype, numpy.number):
        # Numerical feature: create a scatter plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y_true,
        ))
    else:
        # Non-numerical feature: create a box plot
        fig.add_trace(go.Box(
            x=x,
            y=y_true,
            name="Distribution",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8
        ))

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
