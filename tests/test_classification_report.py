import pandas as pd
import polars as pl
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from autofeat.solver import SOLVERS, Problem, Solution, Solver

# Import your BinaryClassificationAnalysis class
from autofeat.ui.binary_classification_analysis import BinaryClassificationAnalysis


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names) # type: ignore
    y = pd.Series(data.target, name="target") # type: ignore
    return X, y

@st.cache_resource
def create_solution(X: pd.DataFrame, y: pd.Series, solver: Solver) -> Solution:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert pandas to polars
    X_train_pl = pl.DataFrame(X_train)
    X_test_pl = pl.DataFrame(X_test)
    y_train_pl = pl.Series(y_train)
    y_test_pl = pl.Series(y_test)

    model = solver.factory()
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

    # Let user select a solver
    solver_names = [solver.name for solver in SOLVERS if solver.problem == Problem.classification]
    selected_solver_name = st.selectbox("Select a model", solver_names)
    selected_solver = next(solver for solver in SOLVERS if solver.name == selected_solver_name and solver.problem == Problem.classification) # noqa: E501

    solution = create_solution(X, y, selected_solver)
    analysis = BinaryClassificationAnalysis(solution)
    analysis.run()
