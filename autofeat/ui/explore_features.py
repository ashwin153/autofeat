from typing import Any

import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
import streamlit

from autofeat.model import Model, PredictionProblem


def explore_features(
    model: Model,
) -> None:
    """Explore the input features to a model.

    :param model: Model to explore.
    """
    _create_feature_importance_charts(model)
    _create_feature_analysis_charts(model)


@streamlit.fragment
def _create_feature_importance_charts(
    model: Model,
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
    model: Model,
) -> None:
    feature_importance = _feature_importance(model)
    feature_importance = feature_importance.sort_values("Importance", ascending=False)
    feature_list = feature_importance["Feature"].tolist()

    # Initialize session state for tabs
    if "tabs" not in streamlit.session_state:
        streamlit.session_state.tabs = []
    else:
        streamlit.session_state.tabs = [tab for tab in streamlit.session_state.tabs if tab["feature"] in feature_list]  #noqa

    tabs_list = streamlit.session_state.tabs
    tab_labels = ["Create Chart"] + [tab["label"] for tab in tabs_list]  # 'New Tab' is now first

    # Use streamlit.tabs to create tabbed interface

    with streamlit.container(border=True):
        tabs = streamlit.tabs(tab_labels)
        # Handle the 'New Tab' for adding new tabs
        with tabs[0]:  # Index 0 for 'New Tab'
            available_features = [
                f for f in feature_list
                if f not in [t.get("feature") for t in streamlit.session_state.tabs if t.get("feature")] #noqa 501
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
                    for chart_fig in _create_classification_feature_chart(model, selected_feature):
                        streamlit.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})  # noqa: E501
                elif model.prediction_method.problem == PredictionProblem.regression:
                    streamlit.subheader(f"{selected_feature} vs. {model.y.name}")
                    chart_fig = _create_regression_feature_chart(model, selected_feature)
                    streamlit.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})  # noqa: E501
                else:
                    raise NotImplementedError(f"{model.prediction_method.problem} is not supported")

                # Button to close the tab
                if streamlit.button("Close Tab", key=f"close_{idx}"):
                    del streamlit.session_state.tabs[idx]
                    streamlit.rerun(scope="fragment")


@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=5,
)
def _create_classification_feature_chart(  # type: ignore[no-any-unimported]
    model: Model,
    feature: str,
) -> list[go.Figure]:
    # Get the index of the feature
    i = model.X.columns.index(feature)
    # Clean the data
    x, y_true = _clean_data(model.X_test[:, i], model.y_test)
    list_of_figs = []
    fig = go.Figure()
    # Check if we have any data left after cleaning
    if len(x) == 0:
        list_of_figs.append(fig)
        return list_of_figs

    df = pandas.DataFrame({"feature": x, f"{model.y.name}": y_true})
    # Check if the feature is numerical
    if model.X.schema[feature].is_numeric():
        # Numerical feature: create a histogram with adjustable buckets
        num_buckets = 5
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
                    "BucketStart": bucket["feature"].min(),
                    "BucketEnd": bucket["feature"].max(),
                })

        bucket_df = pandas.DataFrame(bucket_data)

        fig = px.bar(
            bucket_df, x="Bucket", y="Count", color=model.y.name,
            labels={"Bucket": feature, "Count": f"{target_value} count"},
            height=600, width=800,
        )

        fig.update_layout(bargap=0.2, margin={"t": 30})
        fig.update_xaxes(title_text=f"{feature} (Buckets)")
        fig.update_yaxes(title_text=f"{model.y.name} count")
        list_of_figs.append(fig)

        # Add a box and whisker plot for Y vs. X
        fig_two = go.Figure()
        fig_two.add_trace(
            go.Box(
                y=df["feature"],
                x=df[model.y.name],
                name=feature,
                boxpoints="outliers",
            ),
        )

        # Compute whiskers directly from the entire dataset
        lower_whisker, upper_whisker = _compute_whiskers(df["feature"])

        # Calculate the range with some padding
        padding = (upper_whisker - lower_whisker) * 0.05
        y_min = lower_whisker - padding
        y_max = upper_whisker + padding


        # Update layout for fig_two
        fig_two.update_layout(
            xaxis_title=model.y.name,
            yaxis_title=feature,
            yaxis={"range": [y_min, y_max]},
            margin={"t": 30},
        )

        list_of_figs.append(fig_two)
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
            yaxis={"tickformat": ".1f", "range": [0, 100]},
            xaxis={"type": "category", "categoryorder": "total descending"},
            margin={"t": 20},
        )

        list_of_figs.append(fig)

    return list_of_figs


@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=5,
)
def _create_regression_feature_chart(  # type: ignore[no-any-unimported]
    model: Model,
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
                boxpoints=False,
            ),
        )

    # Customize the layout
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title=f"{model.y.name}",
        margin={"t": 30},
    )

    return fig


@streamlit.cache_data(
    hash_funcs={Model: id},
    max_entries=1,
)
def _feature_importance(
    model: Model,
) -> pandas.DataFrame:
    importance = (
        numpy.abs(model.explanation.values).mean((0, 2))
        if len(model.explanation.shape) == 3
        else numpy.abs(model.explanation.values).mean(0)
    )

    return pandas.DataFrame(
        {
            "Feature": model.X.columns,
            "Importance": importance,
        },
    )


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


@streamlit.cache_data(
    hash_funcs={go.Figure: id},
    max_entries=1,
)
def _compute_whiskers(
    data: pandas.Series,
) -> tuple:
    # Convert to pandas Series to handle mixed types and easily remove NaN/None
    # Remove NaN and None values
    data_clean = data.dropna()

    # Check if we have any data left after removing NaN/None
    if len(data_clean) == 0:
        return numpy.nan, numpy.nan  # Return NaN if no valid data

    # Calculate quartiles and IQR
    q1 = numpy.percentile(data_clean, 25)
    q3 = numpy.percentile(data_clean, 75)
    iqr = q3 - q1

    # Calculate whiskers
    lower_whisker = max(data_clean.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data_clean.max(), q3 + 1.5 * iqr)

    return lower_whisker, upper_whisker
