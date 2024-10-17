import numpy
import pandas
import plotly.graph_objects as go
import streamlit

from autofeat.model import Model
from autofeat.problem import Problem


def combine_features(
    model: Model,
) -> None:
    """Combine features of a model.

    :param model: Model to explore.
    """
    with streamlit.container(border=True):
        streamlit.subheader("Combine Predictors")
        streamlit.caption(f"Set thresholds or values for different predictors to see how they change the value of {model.y.name}") #noqa 501
        feature_list = model.X.columns
        with streamlit.container(border=True):
            # Initialize session state for selected features and rules
            if "selected_features" not in streamlit.session_state:
                streamlit.session_state.selected_features = []
            if "rules" not in streamlit.session_state:
                streamlit.session_state.rules = {}
            else:
                # Remove features from session state that are not in all_features
                streamlit.session_state.selected_features = [f for f in streamlit.session_state.selected_features if f in feature_list] #noqa
                streamlit.session_state.rules = {k: v for k, v in streamlit.session_state.rules.items() if k in feature_list} #noqa

            # Function to update rules when a new feature is selected
            def update_selected_features() -> None:
                new_feature = streamlit.session_state.feature_selector
                if new_feature and new_feature not in streamlit.session_state.selected_features:
                    streamlit.session_state.selected_features.append(new_feature)
                    streamlit.session_state.rules[new_feature] = {"operator": "<", "value": 0.0}

            # Function to update the rule value when the input changes
            def update_rule_value(feature) -> None:  # type: ignore[no-untyped-def]
                streamlit.session_state.rules[feature]["value"] = streamlit.session_state[f"val_{feature}"] #noqa

            streamlit.selectbox(
                "Add a feature to the rules",
                [f for f in feature_list if f not in streamlit.session_state.selected_features],
                key="feature_selector", index=None, on_change=update_selected_features,
            )

            # Display and edit rules for selected features
            for i, feature in enumerate(streamlit.session_state.selected_features):
                if i > 0:
                    streamlit.markdown("---")  # Add separator between rules

                cols = streamlit.columns([3, 1, 2, 0.5])

                with cols[0]:
                    streamlit.markdown(f"**{feature}**")

                if model.X.schema[feature].is_numeric():
                    with cols[1]:
                        streamlit.session_state.rules[feature]["operator"] = streamlit.selectbox(
                            " ", ["<", "<=", ">=", ">"],
                            key=f"op_{feature}",
                            index=["<", "<=", ">=", ">"].index(streamlit.session_state.rules[feature]['operator']), #noqa
                            label_visibility="collapsed",
                        )
                    with cols[2]:
                        #streamlit.session_state.rules[feature]["value"] = streamlit.number_input(
                        streamlit.number_input(
                            "Value", value=streamlit.session_state.rules[feature]['value'], key=f"val_{feature}", #noqa
                            label_visibility="collapsed",
                            step=None,  # Setting step to None still shows the stepper, but allows free input #noqa
                            format="%.3f",  # Allows users to input numbers with high precision
                            on_change=update_rule_value,  # Trigger update when value changes
                            args=(feature,),  # Pass the current feature to the callback
                        )
                else:
                    with cols[2]:
                        unique_values = numpy.unique(model.X_test[:, model.X.columns.index(feature)]) #noqa
                        if "value" not in streamlit.session_state.rules[feature]:
                            streamlit.session_state.rules[feature]["value"] = []
                        streamlit.session_state.rules[feature]["value"] = streamlit.multiselect(
                            "Select values", unique_values,
                            default=streamlit.session_state.rules[feature]["value"],
                            key=f"val_{feature}",
                            label_visibility="collapsed",
                        )

                with cols[3]:
                    if streamlit.button("X", key=f"remove_{feature}"):
                        streamlit.session_state.selected_features.remove(feature)
                        del streamlit.session_state.rules[feature]
                        streamlit.rerun(scope="fragment")

        # Apply rules and calculate statistics
        if streamlit.session_state.rules:
            mask = numpy.ones(len(model.X_test), dtype=bool)
            for feature, rule in streamlit.session_state.rules.items():
                feature_index = model.X.columns.index(feature)
                feature_values = model.X_test[:, feature_index]
                if model.X.schema[feature].is_numeric():
                    operator = rule["operator"]
                    value = rule["value"]
                    if operator == "<":
                        mask &= (feature_values < value)
                    elif operator == "<=":
                        mask &= (feature_values <= value)
                    elif operator == ">=":
                        mask &= (feature_values >= value)
                    elif operator == ">":
                        mask &= (feature_values > value)
                else:
                    mask &= numpy.isin(feature_values, rule["value"])

            # Calculate and display statistics
            group_size = mask.sum()
            total_size = len(mask)
            group_percentage = group_size / total_size * 100

            y_true = model.y_test
            match model.prediction_method.problem:
                case Problem.classification:
                    unique_classes = numpy.unique(y_true)
                    class_stats = []

                    for cls in unique_classes:
                        group_class_count = (y_true[mask] == cls).sum()
                        group_class_percentage = group_class_count / group_size * 100 if group_size > 0 else 0  #noqa

                        total_class_count = (y_true == cls).sum()
                        class_recall = group_class_count / total_class_count * 100

                        class_stats.append({
                            "Class": cls,
                            "Prevalence": group_class_percentage,
                            "Recall": class_recall,
                        })

                    # Extract classes and prevalence from class_stats
                    classes = [stat["Class"] for stat in class_stats]
                    prevalences = [stat["Prevalence"] for stat in class_stats]

                    # Create the bar chart
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=classes,
                                y=prevalences,
                                text=[f"{prev:.2f}%" for prev in prevalences],
                                textposition="auto",
                            ),
                        ],
                    )

                    # Update layout
                    fig.update_layout(
                        title=f"{model.y.name} Prevalence in Selected Group",
                        xaxis_title=f"{model.y.name}",
                        yaxis_title= "% of selected group",
                        yaxis={"range": [0, 100]},
                    )
                    streamlit.plotly_chart(fig, use_container_width=True)

                    # Create and display the dataframe with formatted percentages and renamed columns #noqa
                    df = pandas.DataFrame(class_stats)
                    df = df.rename(
                        columns={
                            "Class": f"{model.y.name}",
                            "Prevalence": "Prevalence in Selected Group",
                            "Recall": f"Coverage of {model.y.name} value",
                        },
                    )
                    df["Prevalence in Selected Group"] = df["Prevalence in Selected Group"].apply(lambda x: f"{x:.2f}%") #noqa
                    df[f"Coverage of {model.y.name} value"] = df[f"Coverage of {model.y.name} value"].apply(lambda x: f"{x:.2f}%")  #noqa

                    streamlit.dataframe(df, hide_index=True, use_container_width=True)
                    streamlit.write(f"Selected group size: {group_size} ({group_percentage:.2f}% of total)")  #noqa

                case Problem.regression:
                    # 1. Create a box plot for the selected group vs. the rest
                    selected_y_values = y_true[mask]
                    rest_y_values = y_true[~mask]

                    fig = go.Figure()
                    fig.add_trace(go.Box(y=selected_y_values, name='Selected Group', marker_color='blue', boxpoints=False)) #noqa
                    fig.add_trace(go.Box(y=rest_y_values, name='Rest of Population', marker_color='orange', boxpoints=False)) #noqa

                    # Update layout for the box plot
                    fig.update_layout(
                        title=f"{model.y.name} Distribution: Selected Group vs Rest",
                        yaxis_title=f"{model.y.name}",
                    )
                    streamlit.plotly_chart(fig, use_container_width=True)

                    # 2. Create a table comparing mean and median of the two groups
                    selected_mean = numpy.mean(selected_y_values)
                    selected_median = numpy.median(selected_y_values)
                    rest_mean = numpy.mean(rest_y_values)
                    rest_median = numpy.median(rest_y_values)

                    mean_diff = ((selected_mean - rest_mean) / rest_mean) * 100 if rest_mean != 0 else 0  #noqa
                    median_diff = ((selected_median - rest_median) / rest_median) * 100 if rest_median != 0 else 0  #noqa

                    comparison_df = pandas.DataFrame({
                        "Group": ["Selected Group", "Rest of Population"],
                        "Mean": [selected_mean, rest_mean],
                        "Median": [selected_median, rest_median],
                    })
                    # Format the numeric columns to two decimal points
                    comparison_df["Mean"] = comparison_df["Mean"].apply(lambda x: f"{x:.2f}")
                    comparison_df["Median"] = comparison_df["Median"].apply(lambda x: f"{x:.2f}")
                    # Append the percentage difference row
                    diff_row = pandas.DataFrame({
                        "Group": ["% Difference"],
                        "Mean": [f"{mean_diff:.2f}%"],
                        "Median": [f"{median_diff:.2f}%"],
                    })
                    comparison_df = pandas.concat([comparison_df, diff_row], ignore_index=True)

                    # Display the comparison table
                    streamlit.dataframe(comparison_df, hide_index=True, use_container_width=True)

                    # 3. Describe the size of the isolated group
                    streamlit.write(f"Selected group size: {group_size} ({group_percentage:.2f}% of total)") #noqa
        else:
            streamlit.info("Select features and set rules to see statistics.")
