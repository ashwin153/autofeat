from typing import Any

import numpy
import polars
import sklearn.metrics
import streamlit

from autofeat.model import Model, PredictionProblem


def evaluate_model(
    model: Model,
) -> None:
    """Evaluate the performance of the model.

    :param model: Model to evaluate.
    """
    with streamlit.container(border=True):
        streamlit.subheader("Model Performance")

        streamlit.markdown(_headline(model))

        with streamlit.expander("Show detailed model stats"):
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

            streamlit.caption(_caption(model))


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

    return f"✅ Model is **:green[{metric['Improvement']:.2f}% better]** than {_baseline(model)}"


def _baseline(
    model: Model,
) -> str:
    match model.prediction_method.problem:
        case PredictionProblem.classification:
            return "always guessing the most frequent category"
        case PredictionProblem.regression:
            return "always guessing mean"
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


def _primary_metric(
    model: Model,
) -> str:
    match model.prediction_method.problem:
        case PredictionProblem.classification:
            return "F1"
        case PredictionProblem.regression:
            return "R2"
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
        case PredictionProblem.classification:
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
        case PredictionProblem.regression:
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
                    _percent_change(baseline["RMSE"], metrics["RMSE"]),
                    _percent_change(baseline["R2"],metrics["R2"]),
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


@streamlit.cache_data(
    hash_funcs={
        Model: id,
    },
)
def _caption(
    model: Model,
) -> str:
    match model.prediction_method.problem:
        case PredictionProblem.classification:
            return (
                "Comparison: model's performance against a baseline model that randomly guesses "
                f"based on the frequency of {model.y.name} values. Higher precision indicates a "
                "model guesses the value correctly more on average. Higher recall indicates that a "
                "model covers more correct classifications overall."
            )
        case PredictionProblem.regression:
            return (
                "Comparison: the model's performance against a baseline model, that always "
                f"predicts the mean of {model.y.name}. RMSE is the average error between the "
                "actual value and the prediction by the model. Lower error is better (means "
                "guesses are closer to true). R2 indicates how much of the variation in your data "
                "the model captures. A higher value is better."
            )
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")
