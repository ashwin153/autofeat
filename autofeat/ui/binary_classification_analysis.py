import altair as alt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import precision_score, recall_score
from streamlit_shap import st_shap


class BinaryClassificationAnalysis:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = self.get_feature_names()
        self.y_pred_proba = self.model.predict_proba(self.X)[:, 1]

    def get_feature_names(self):
        if hasattr(self.model, "feature_names_in_"):
            return self.model.feature_names_in_
        elif isinstance(self.X, pd.DataFrame):
            return self.X.columns.tolist()
        else:
            return [f"Feature_{i}" for i in range(self.X.shape[1])]

    def run(self):
        st.title("Analysis of model performance and predictors")

        self.section_1_model_performance()
        self.section_2_feature_importance()

    def section_1_model_performance(self):
        st.header("How good is your model?")

        # Allow user to set threshold
        col1, col2 = st.columns([3, 1])
        with col1:
            threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)
        with col2:
            st.metric("Current threshold", f"{threshold:.2f}")


        # Calculate metrics
        predictions = (self.y_pred_proba >= threshold).astype(int)
        current_precision = precision_score(self.y, predictions)
        current_recall = recall_score(self.y, predictions)
        current_coverage = np.mean(self.y_pred_proba >= threshold)
        baseline_precision = np.mean(self.y)

        # Create the data for the table
        data = {
            "Metric": ["Precision", "Recall", "Overall Coverage"],
            "Value": [
                f"{current_precision:.1%}",
                f"{current_recall:.1%}",
                f"{current_coverage:.1%}",
            ],
            "Explanation": [
                f"""If we predict a sample to have a score above this threshold, {current_precision:.1%} likely they will be classified correctly. This is an improvement of +{current_precision - baseline_precision:.1%} over the average proportion of {baseline_precision:.1%}.""", # noqa: E501
                f"""{current_recall:.1%} of all the samples that are classified as 1 are covered above this threshold.""", # noqa: E501
                f"""{current_coverage:.1%} of all samples will have a score above this threshold.""", # noqa: E501
            ],
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Display the table
        st.write("### Model Performance Metrics")

        # Apply custom styling to the table for even columns
        styled_df = df.style.set_properties(
            **{
                "width": f"{100/3}%",
                "text-align": "left",
                "white-space": "pre-wrap",
            },
        )

        st.table(styled_df)
        # Create DataFrame for samples over and under the threshold
        df_results = pd.DataFrame({
            "Target": self.y,
            "Predicted Score": self.y_pred_proba,
            **{name: self.X[name] for name in self.feature_names},
        })

        df_over_threshold = df_results[df_results["Predicted Score"] >= threshold]
        df_under_threshold = df_results[df_results["Predicted Score"] < threshold]

        st.subheader("Detailed Results Tables")

        with st.expander("Samples Over Threshold"):
            st.dataframe(df_over_threshold)  # Display the over-threshold data

        with st.expander("Samples Under Threshold"):
            st.dataframe(df_under_threshold)  # Display the under-threshold data

    def section_2_feature_importance(self):
        st.header("Feature Importance")

        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate feature importance based on mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": feature_importance,
        }).sort_values("Importance", ascending=False)  # Features are sorted here

        features_per_page = 10
        total_features = len(feature_importance_df)
        total_pages = (total_features + features_per_page - 1) // features_per_page

        page = st.number_input("Page number", min_value=1, max_value=total_pages, value=1)

        # Select features for the current page based on overall ranking
        start_idx = (page - 1) * features_per_page
        end_idx = min(start_idx + features_per_page, total_features)
        current_features = feature_importance_df.iloc[start_idx:end_idx]

        st.subheader(f"Top Features by Importance (Page {page}/{total_pages})")

        # Display SHAP values for current features using st_shap
        current_shap_values = shap_values[:, current_features.index]
        current_feature_names = current_features["Feature"].values

        # Create a DataFrame for the current features and their SHAP values
        shap_df = pd.DataFrame(current_shap_values, columns=current_feature_names)

        # Display SHAP summary plot
        st_shap(shap.summary_plot(shap_df.values, feature_names=current_feature_names, plot_type="bar")) # noqa: E501

        # Display individual charts for each feature
        for feature in current_feature_names:
            self.create_chart(feature)

    def create_chart(self, feature):
        st.write(f"#### {feature} Relationship with Target:")

        if self.is_quantitative(self.X, feature):
            # Show a line chart for quantitative features
            st.line_chart(self.X[feature].value_counts())
        else:
            # Show a box plot for categorical features
            st.altair_chart(
                alt.Chart(self.X).mark_boxplot(extent="min-max")
                .encode(x=feature, y="target_variable"),
                use_container_width=True,
            )

    def is_quantitative(self, df, feature):
        return df[feature].dtype in ["int64", "float64"]


def run_analysis(model, X, y):
    st.set_page_config(layout="wide")
    analysis = BinaryClassificationAnalysis(model, X, y)
    analysis.run()
