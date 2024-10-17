from typing import Any

import numpy
import polars
import sklearn.metrics
import streamlit

from autofeat.model import Model
from autofeat.problem import Problem


def evaluate_model(
    model: Model,
) -> None:
    """Evaluate the performance of the model.

    :param model: Model to evaluate.
    """
    streamlit.markdown(
        _headline(model),
        help=_caption(model),
    )

    with streamlit.expander("Model Performance"):
        streamlit.dataframe(
            _metrics(model),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model": streamlit.column_config.NumberColumn(format="%.4f"),
                "Baseline": streamlit.column_config.NumberColumn(format="%.4f"),
                "Improvement": streamlit.column_config.NumberColumn(format="%.2f %%"),
            },
        )


@streamlit.cache_data(
    hash_funcs={
        Model: id,
    },
)
def _headline(
    model: Model,
) -> str:
    metric = (
        _metrics(model)
        .filter(polars.col("Metric") == _primary_metric(model))
        .row(0, named=True)
    )

    return (
        f"✅ Your model is **:green[{metric['Improvement']:.2f}% better]** than baseline "
        "performance"
    )


def _caption(
    model: Model,
) -> str:
    match model.prediction_method.problem:
        case Problem.classification:
            return (
                "Classification models are benchmarked against a baseline model that always "
                "guesses the most frequently occurring, and improvement is measured in F1 scores. "
                "F1 is a measure of the accuracy of a model's predictions. Higher F1 is better."
            )
        case Problem.regression:
            return (
                "Regression models are benchmarked against a baseline model that always guesses "
                "the mean, and improvement is measured in root mean squared error (RMSE). RMSE "
                "measures the distance between a model's predictions to their true values. Lower "
                "RMSE is better."
            )


def _primary_metric(
    model: Model,
) -> str:
    match model.prediction_method.problem:
        case Problem.classification:
            return "F1"
        case Problem.regression:
            return "RMSE"
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


@streamlit.cache_data(
    hash_funcs={
        Model: id,
    },
)
def _metrics(
    model: Model,
) -> polars.DataFrame:
    match model.prediction_method.problem:
        case Problem.classification:
            metrics = _classification_metrics(model.y_test, model.y_predicted)
            baseline = _classification_metrics(model.y_test, model.y_baseline)

            return polars.DataFrame({
                "Metric": [
                    "Accuracy",
                    "F1",
                    "Precision",
                    "Recall",
                ],
                "Model": [
                    metrics["Accuracy"],
                    metrics["F1"],
                    metrics["Precision"],
                    metrics["Recall"],
                ],
                "Baseline": [
                    baseline["Accuracy"],
                    baseline["F1"],
                    baseline["Precision"],
                    baseline["Recall"],
                ],
                "Improvement": [
                    _percent_change(baseline["Accuracy"], metrics["Accuracy"]),
                    _percent_change(baseline["F1"], metrics["F1"]),
                    _percent_change(baseline["Precision"], metrics["Precision"]),
                    _percent_change(baseline["Recall"], metrics["Recall"]),
                ],
            })
        case Problem.regression:
            metrics = _regression_metrics(model.y_test, model.y_predicted)
            baseline = _regression_metrics(model.y_test, model.y_baseline)

            return polars.DataFrame({
                "Metric": [
                    "RMSE",
                    "R2",
                ],
                "Model": [
                    metrics["RMSE"],
                    metrics["R2"],
                ],
                "Baseline": [
                    baseline["RMSE"],
                    baseline["R2"],
                ],
                "Improvement": [
                    -_percent_change(baseline["RMSE"], metrics["RMSE"]),
                    abs(_percent_change(baseline["R2"],metrics["R2"])),
                ],
            })
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


def _classification_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
) -> dict[str, Any]:
    return {
        "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "F1": sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
        "Precision": sklearn.metrics.precision_score(y_true, y_pred, average="macro"),
        "Recall": sklearn.metrics.recall_score(y_true, y_pred, average="macro"),
    }


def _regression_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
) -> dict[str, Any]:
    return {
        "R2": sklearn.metrics.r2_score(y_true, y_pred),
        "RMSE": sklearn.metrics.root_mean_squared_error(y_true, y_pred),
    }


def _percent_change(
    old: float,
    new: float,
) -> float:
    return (new - old) / old * 100
