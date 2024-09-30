import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

from autofeat.solver import Solution


class RegressionAnalysis:
    def __init__(self, solution: Solution) -> None:
        self.solution = solution
        self.feature_names = self.get_feature_names()
        self.y_pred = self.solution.model.predict(self.solution.X_test.to_numpy())

    def get_feature_names(self) -> list[str]:
        return self.solution.X_test.columns

    def run(self) -> None:
        self.section_0_summary()

    def section_0_summary(self) -> None:
        # Calculate model performance
        mse = mean_squared_error(self.solution.y_test.to_numpy(), self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.solution.y_test.to_numpy(), self.y_pred)

        # Calculate baseline performance (using mean prediction)
        y_mean = np.mean(self.solution.y_train.to_numpy())
        baseline_mse = mean_squared_error(self.solution.y_test.to_numpy(), [y_mean] * len(self.y_pred)) # noqa: E501
        baseline_rmse = np.sqrt(baseline_mse)

        improvement = baseline_rmse / rmse

        st.header("Model & predictors of target successfully generated")

        # Green shaded box with dropdown
        st.markdown(
            """
            <style>
            .stExpander {
                border: 1px solid #28a745;
                border-radius: 0.5rem;
                padding: 1rem;
                background-color: #d4edda;
            }
            .stExpander > div:first-child {
                background-color: #d4edda !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"#### ✅ Model is {improvement:.2f}x more accurate than the baseline", expanded=False): # noqa: E501
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"• Model RMSE: {rmse:.4f}")
            st.write(f"• Baseline RMSE (mean prediction): {baseline_rmse:.4f}")
            st.write(f"• R-squared score: {r2:.4f}")

        # Feature importance section
        self.paginated_feature_importance()

    def paginated_feature_importance(self) -> None:
        shap_values = self.solution.shap_values
        feature_importance = np.abs(shap_values.values).mean(0)

        feature_importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": feature_importance,
        }).sort_values("Importance", ascending=False)

        features_per_page = 8
        total_features = len(feature_importance_df)
        total_pages = max(1, (total_features + features_per_page - 1) // features_per_page)

        trimmed_chart = st.container()

        # Pagination using number input
        _, col_pagination = st.columns([3,1])
        with col_pagination:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="feature_importance_page",
            )
        # Update chart based on current page
        start_idx = (current_page - 1) * features_per_page
        end_idx = min(start_idx + features_per_page, total_features)
        current_features = feature_importance_df.iloc[start_idx:end_idx]

        updated_chart = alt.Chart(current_features).mark_bar().encode(
            y=alt.Y("Feature:N", sort="-x", title=None),
            x=alt.X("Importance:Q", scale=alt.Scale(domain=[0, feature_importance_df["Importance"].max()])), # noqa: E501
            tooltip=["Feature", "Importance"],
        ).properties(
            title="Top Predictors of Target",
        )

        trimmed_chart.altair_chart(updated_chart, use_container_width=True)
