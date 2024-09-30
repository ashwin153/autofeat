# import streamlit

# from autofeat.dataset import Dataset


# def column_selector(
#     dataset: Dataset,
# ) -> TODO:
#     training_data = streamlit.selectbox(
#         "Training Data",
#         [table.name for table in dataset.tables],
#     )

#     if training_data is None:
#         return

#     table = dataset.table(training_data)

#     target_column = streamlit.selectbox(
#         "Target Column",
#         table.schema,
#     )

#     if target_column is None:
#         return

#     known_columns = streamlit.multiselect(
#         "Known Columns",
#         set(table.schema) - {target_column},
#     )

#     if known_columns is None:
#         return

#     problem = streamlit.selectbox(
#         "Problem",
#         ("Regression", "Classification"),
#         index=0 if Attribute.numeric in table.schema[target_column] else 1,
#     )

#     if problem == "Regression":
#         model = streamlit.selectbox(
#             "Classification Model",
#             ("XGBoost", "Random Forest"),
#         )
#     elif problem == "Classification":
#         model = streamlit.selectbox(
#             "Regression Model",
#             ("XGBoost", "Random Forest"),
#         )

#         if model ==
