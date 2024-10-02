import math
from typing import Any

import altair
import numpy
import pandas
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
                f"✅ model is {improvement:.2f}% more accurate than random guessing",
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
                f"✅ model is {improvement:.2f}% more accurate than linear regression",
            ):
                streamlit.metric(
                    "RMSE",
                    value=f"{metrics['rmse']:.4f}",
                    delta=f"{_percent_change(baseline['accuracy'], metrics['accuracy']):.2f}%",
                )

                streamlit.metric(
                    "R2",
                    value=f"{metrics['r2']:.4f}",
                    delta=f"{_percent_change(baseline['r2'], metrics['r2']):.2f}%",
                )
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

    streamlit.altair_chart(
        (
            altair.Chart(_feature_importance(model))
            .mark_bar()
            .configure_axis(labelLimit=300)
            .encode(y=altair.Y("Feature:N", sort="-x"), x=altair.X("Importance:Q"))
        ),
        use_container_width=True,
    )


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
def _feature_importance(
    model: TrainedModel,
) -> pandas.DataFrame:
    shap_explainer = shap.Explainer(model.prediction_model)
    shap_explanation = shap_explainer(model.X_test)

    return pandas.DataFrame(
        {
            "Feature": model.X.columns,
            "Importance": numpy.abs(shap_explanation.values).mean(0),
        },
    )


def _percent_change(
    old: float,
    new: float,
) -> float:
    return (new - old) / old * 100
