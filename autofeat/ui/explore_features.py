import math

import numpy
import pandas
import plotly.express
import plotly.graph_objects
import streamlit
import streamlit_theme

from autofeat.model import Model, PredictionProblem


@streamlit.fragment
def explore_features(
    model: Model,
) -> None:
    """Explore the input features to a model.

    :param model: Model to explore.
    """
    df = _grid(model)
    plotly_theme = _plotly_theme()

    column1, column2 = streamlit.columns(2)

    with column1:
        event = streamlit.dataframe(
            df,
            column_config={
                "Feature": None,
                "Importance": streamlit.column_config.ProgressColumn(
                    format="%.2f",
                    max_value=1,
                    min_value=0,
                ),
            },
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

    with column2:
        for row in event.get("selection", {}).get("rows", []):
            for fig in _charts(model, df.iloc[row]["Feature"], plotly_theme):
                streamlit.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )


def _plotly_theme() -> str:
    theme = streamlit_theme.st_theme()

    return (
        "plotly"
        if theme is None or theme["base"] == "light"
        else "plotly_dark"
    )


@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=1,
)
def _grid(
    model: Model,
) -> pandas.DataFrame:
    importance = (
        numpy.abs(model.explanation.values).mean((0, 2))
        if len(model.explanation.shape) == 3
        else numpy.abs(model.explanation.values).mean(0)
    )

    importance = importance / numpy.max(importance)

    df = pandas.DataFrame({
        "Feature": model.X.columns,
        "Predictor": [c.split(" :: ", 1)[0] for c in model.X.columns],
        "Importance": importance,
        "Source": [c.split(" :: ", 1)[1] for c in model.X.columns],
    })

    df = df.sort_values("Importance", ascending=False)

    return df


@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=10,
)
def _charts(  # type: ignore[no-any-unimported]
    model: Model,
    feature: str,
    plotly_theme: str,
) -> list[plotly.graph_objects.Figure]:
    x, y = model.X_test[:, model.X.columns.index(feature)], model.y_test
    mask = ~(pandas.isnull(x) | pandas.isnull(y))
    x, y = x[mask], y[mask]

    df = pandas.DataFrame({"x": x, "y": y})
    df = df.sort_values("x")

    df.attrs = {
        "x_label": feature,
        "y_label": model.y.name,
        "plotly_theme": plotly_theme,
    }

    match model.prediction_method.problem:
        case PredictionProblem.classification:
            if model.X.schema[feature].is_numeric():
                return [_histogram(df), _box_and_whisker_plot(df)]
            else:
                return [_stacked_bar_chart(df)]
        case PredictionProblem.regression:
            if model.X.schema[feature].is_numeric():
                return [_scatter_plot(df)]
            else:
                return [_box_plot(df)]
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


def _histogram(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> plotly.graph_objects.Figure:
    buckets = []
    num_buckets = 5
    bucket_size = math.ceil(len(df) / num_buckets)
    categories = df["y"].unique()

    for i in range(num_buckets):
        bucket_data = df.iloc[i*bucket_size:(i+1)*bucket_size]
        bucket_name = f"{bucket_data['x'].min():.2f} - {bucket_data['x'].max():.2f}"

        for category in categories:
            buckets.append({
                "x": bucket_name,
                "category": category,
                "y": len(bucket_data[bucket_data["y"] == category]),
            })

    fig = plotly.express.bar(
        pandas.DataFrame(buckets),
        x="x",
        y="y",
        color="category",
        template=df.attrs["plotly_theme"],
        labels={"x": df.attrs["x_label"], "y": df.attrs["y_label"]},
        height=600, width=800,
    )

    fig.update_layout(
        bargap=0.2,
        margin={"t": 30},
        template=df.attrs["plotly_theme"],
    )

    fig.update_xaxes(title_text=df.attrs["x_label"])
    fig.update_yaxes(title_text=f"count({df.attrs['y_label']})")

    return fig


def _box_and_whisker_plot(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> plotly.graph_objects.Figure:
    fig = plotly.graph_objects.Figure()

    fig.add_trace(
        plotly.graph_objects.Box(
            boxpoints="outliers",
            name=df.attrs["x_label"],
            x=df["y"],
            y=df["x"],
        ),
    )

    q1 = df["x"].quantile(0.25)
    q3 = df["x"].quantile(0.75)
    iqr =  q3 - q1

    lower_whisker = max(df["x"].min(), q1 - 1.5 * iqr)
    upper_whisker = min(df["x"].max(), q3 + 1.5 * iqr)
    padding = (upper_whisker - lower_whisker) * 0.05

    y_min = lower_whisker - padding
    y_max = upper_whisker + padding

    fig.update_layout(
        margin={"t": 30},
        template=df.attrs["plotly_theme"],
        xaxis_title=df.attrs["y_label"],
        yaxis_title=df.attrs["x_label"],
        yaxis={"range": [y_min, y_max]},
    )

    return fig


def _stacked_bar_chart(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> plotly.graph_objects.Figure:
    counts = (
        df
        .groupby(["x", "y"])
        .size()
        .unstack(fill_value=0)
    )

    percentages = counts.apply(lambda x: x / x.sum() * 100, axis=1)

    fig = plotly.express.bar(
        percentages,
        barmode="stack",
        labels={
            "x": df.attrs["x_label"],
            "y": "Percentage",
            "color": df.attrs["y_label"],
        },
        template=df.attrs["plotly_theme"],
        x=percentages.index,
        y=percentages.columns,
    )

    fig.update_layout(
        margin={"t": 20},
        template=df.attrs["plotly_theme"],
        xaxis_title=df.attrs["x_label"],
        xaxis={"type": "category", "categoryorder": "total descending"},
        yaxis_title=f"{df.attrs['y_label']} (%)",
        yaxis={"tickformat": ".1f", "range": [0, 100]},
    )

    return fig


def _scatter_plot(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> plotly.graph_objects.Figure:
    fig = plotly.graph_objects.Figure()

    fig.add_trace(
        plotly.graph_objects.Scatter(
            marker={
                "size": 5,
                "opacity": 0.6,
            },
            mode="markers",
            name=df.attrs["y_label"],
            x=df["x"],
            y=df["y"],
        ),
    )

    x = pandas.to_numeric(df["x"], errors="coerce")
    best_fit_coeffs = numpy.polyfit(x, df["y"], 1)
    best_fit_x = numpy.array([numpy.min(x), numpy.max(x)])
    best_fit_y = best_fit_coeffs[0] * best_fit_x + best_fit_coeffs[1]

    fig.add_trace(
        plotly.graph_objects.Scatter(
            mode="lines",
            name="Line of Best Fit",
            x=best_fit_x,
            y=best_fit_y,
        ),
    )

    fig.update_layout(
        template=df.attrs["plotly_theme"],
        xaxis_title=df.attrs["x_label"],
        yaxis_title=df.attrs["y_label"],
        margin={"t": 30},
    )

    return fig


def _box_plot(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> plotly.graph_objects.Figure:
    fig = plotly.graph_objects.Figure()

    fig.add_trace(
        plotly.graph_objects.Box(
            boxpoints=False,
            name="Distribution",
            x=df["x"],
            y=df["y"],
        ),
    )

    fig.update_layout(
        template=df.attrs["plotly_theme"],
        xaxis_title=df.attrs["x_label"],
        yaxis_title=df.attrs["y_label"],
        margin={"t": 30},
    )

    return fig
