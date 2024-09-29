import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
@st.cache_resource
def create_and_train_model(X: pd.DataFrame, y: pd.Series) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]: # noqa: E501
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    return model, X_test, y_test  # Return test data instead of train data

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    X, y = load_data()
    model, X_test, y_test = create_and_train_model(X, y)
    analysis = BinaryClassificationAnalysis(model, X_test, y_test)
    analysis.run()
