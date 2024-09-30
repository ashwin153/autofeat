import pandas as pd
import polars as pl
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from autofeat.solver import Problem, Solution, Solver
from autofeat.ui.regression_analysis import RegressionAnalysis


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # Load the California Housing dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

@st.cache_resource
def create_solution(X: pd.DataFrame, y: pd.Series) -> Solution:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert pandas to polars
    X_train_pl = pl.DataFrame(X_train)
    X_test_pl = pl.DataFrame(X_test)
    y_train_pl = pl.Series(y_train)
    y_test_pl = pl.Series(y_test)

    # Use CatBoost regressor
    solver = Solver(
        factory=CatBoostRegressor,
        name="CatBoost",
        problem=Problem.regression,
    )

    model = solver.factory(verbose=False)  # Set verbose=False to suppress CatBoost output
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = model.predict(X_test.to_numpy())

    return Solution(
        model=model,
        problem=solver.problem,
        X_test=X_test_pl,
        X_train=X_train_pl,
        y_test=y_test_pl,
        y_train=y_train_pl,
        y_pred=pl.Series(y_pred),
    )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    X, y = load_data()

    solution = create_solution(X, y)
    analysis = RegressionAnalysis(solution)
    analysis.run()
