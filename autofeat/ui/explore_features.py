import math

import numpy
import pandas
import plotly.express
import plotly.graph_objects
import plotly.subplots
import streamlit

from autofeat.model import Model, PredictionProblem
from autofeat.transform.extract import Extract
from autofeat.ui.edit_settings import Settings


@streamlit.fragment
def explore_features(
    model: Model,
    settings: Settings,
) -> None:
    """Explore the input features to a model.

    :param model: Model to explore.
    """
    df = _grid(model)

    column1, column2 = streamlit.columns(2)

    with column1:
        on_select = streamlit.dataframe(
            df,
            column_config={
                "Id": None,
                "Importance": streamlit.column_config.ProgressColumn(
                    format="%.2f",
                    max_value=1,
                    min_value=0,
                ),
            },
            height=505,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

    with column2:
        if rows := on_select.get("selection", {}).get("rows", []):
            assert len(rows) == 1

            charts = _charts(
                model,
                df.iloc[rows[0]]["Id"],
                settings,
            )

            for figure, tab in zip(
                [figure for _, figure in charts],
                streamlit.tabs([chart for chart, _ in charts]),
            ):
                with tab:
                    streamlit.plotly_chart(
                        figure,
                        use_container_width=True,
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
        "Id": model.X.columns,
        "Feature": [c.split(Extract.SEPARATOR, 1)[0] for c in model.X.columns],
        "Importance": importance,
        "Source": [c.split(Extract.SEPARATOR, 1)[1] for c in model.X.columns],
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
    settings: Settings,
) -> list[tuple[str, plotly.graph_objects.Figure]]:
    x, y = model.X_test[:, model.X.columns.index(feature)], model.y_test
    mask = ~(pandas.isnull(x) | pandas.isnull(y))
    x, y = x[mask], y[mask]

    df = pandas.DataFrame({"x": x, "y": y}).sort_values("x")
    df_flip = pandas.DataFrame({"x": y, "y": x}).sort_values("x")

    df.attrs = {
        "x_label": feature.split(Extract.SEPARATOR, 1)[0],
        "y_label": model.y.name,
        "chart_theme": settings.chart_theme,
    }

    df_flip.attrs = {
        **df.attrs,
        "x_label": df.attrs["y_label"],
        "y_label": df.attrs["x_label"],
    }

    match model.prediction_method.problem:
        case PredictionProblem.classification:
            if model.X.schema[feature].is_numeric():
                return [_histogram(df), _box_plot(df_flip)]
            else:
                return [_stacked_bar_chart(df), _pie_chart(df)]
        case PredictionProblem.regression:
            if model.X.schema[feature].is_numeric():
                return [_scatter_plot(df)]
            else:
                return [_box_plot(df)]
        case _:
            raise NotImplementedError(f"{model.prediction_method.problem} is not supported")


def _box_plot(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> tuple[str, plotly.graph_objects.Figure]:
    figure = plotly.graph_objects.Figure()

    figure.add_trace(
        plotly.graph_objects.Box(
            boxpoints="outliers",
            name="Distribution",
            x=df["x"],
            y=df["y"],
        ),
    )

    q1 = df["y"].quantile(0.25)
    q3 = df["y"].quantile(0.75)
    iqr =  q3 - q1

    y_min = max(df["y"].min(), q1 - 1.5 * iqr)
    y_max = min(df["y"].max(), q3 + 1.5 * iqr)

    figure.update_layout(
        margin={"t": 30},
        template=df.attrs["chart_theme"],
        xaxis_title=df.attrs["x_label"],
        yaxis_title=df.attrs["y_label"],
        yaxis={"range": [y_min, y_max]},
    )

    return ":material/indeterminate_check_box:", figure


def _histogram(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> tuple[str, plotly.graph_objects.Figure]:
    buckets = []
    num_buckets = 5
    bucket_size = math.ceil(len(df) / num_buckets)
    categories = df["y"].unique()

    for i in range(num_buckets):
        bucket_data = df.iloc[i*bucket_size:(i+1)*bucket_size]
        bucket_name = f"{bucket_data['x'].min():.2f} - {bucket_data['x'].max():.2f}"

        total_count = len(bucket_data)

        for category in categories:
            category_count = len(bucket_data[bucket_data["y"] == category])
            percentage = (category_count / total_count) * 100 if total_count > 0 else 0
            buckets.append({
                df.attrs["y_label"]: str(category),
                "x": bucket_name,
                "y": len(bucket_data[bucket_data["y"] == category]),
                "percentage": f"{percentage:.2f}",
            })

    figure = plotly.express.bar(
        pandas.DataFrame(buckets),
        x="x",
        y="y",
        color=df.attrs["y_label"],
        template=df.attrs["chart_theme"],
        labels={"x": df.attrs["x_label"], "y": "count"},
        hover_data=["percentage"],
    )

    figure.update_layout(
        bargap=0.2,
        margin={"t": 30},
    )

    return ":material/equalizer:", figure


def _pie_chart(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> tuple[str, plotly.graph_objects.Figure]:
    figure = plotly.subplots.make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "sunburst"},
                {"type": "sunburst"},
            ],
        ],
    )

    figure.add_trace(
        plotly.express.sunburst(
            df,
            path=["y", "x"],
            template=df.attrs["chart_theme"],
        ).data[0],
        row=1,
        col=1,
    )

    figure.add_trace(
        plotly.express.sunburst(
            df,
            path=["x", "y"],
            template=df.attrs["chart_theme"],
        ).data[0],
        row=1,
        col=2,
    )

    figure.update_traces(
        textinfo="label+percent parent",
    )

    return ":material/pie_chart:", figure


def _scatter_plot(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> tuple[str, plotly.graph_objects.Figure]:
    figure = plotly.graph_objects.Figure()

    figure.add_trace(
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

    figure.add_trace(
        plotly.graph_objects.Scatter(
            mode="lines",
            name="Line of Best Fit",
            x=best_fit_x,
            y=best_fit_y,
        ),
    )

    figure.update_layout(
        template=df.attrs["chart_theme"],
        xaxis_title=df.attrs["x_label"],
        yaxis_title=df.attrs["y_label"],
        margin={"t": 30},
    )

    return ":material/scatter_plot:", figure


def _stacked_bar_chart(  # type: ignore[no-any-unimported]
    df: pandas.DataFrame,
) -> tuple[str, plotly.graph_objects.Figure]:
    counts = (
        df
        .groupby(["x", "y"])
        .size()
        .unstack(fill_value=0)
    )

    percentages = counts.apply(lambda x: x / x.sum() * 100, axis=1)

    figure = plotly.express.bar(
        percentages,
        barmode="stack",
        labels={
            "x": df.attrs["x_label"],
            "y": "Percentage",
            "color": df.attrs["y_label"],
        },
        template=df.attrs["chart_theme"],
        x=percentages.index,
        y=percentages.columns,
    )

    figure.update_layout(
        margin={"t": 20},
        template=df.attrs["chart_theme"],
        xaxis_title=df.attrs["x_label"],
        xaxis={"type": "category", "categoryorder": "total descending"},
        yaxis_title=f"{df.attrs['y_label']} (%)",
        yaxis={"tickformat": ".1f", "range": [0, 100]},
    )

    return ":material/stacked_bar_chart:", figure
