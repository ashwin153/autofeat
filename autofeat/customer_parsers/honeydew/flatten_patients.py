import ast
import json

import pandas as pd


def flatten_medical_background(df):
    """
    Flatten medical background using type field to identify and handle lists
    """
    # Convert medicalbackground to list of dicts
    records = []

    for _, row in df.iterrows():
        if pd.isna(row["medicalbackground"]) or not row["medicalbackground"]:
            records.append({})
            continue

        data = json.loads(row["medicalbackground"])

        # Start with basic fields
        record = {
            "height": data.get("height"),
            "weight": data.get("weight"),
            "sex_medical": data.get("sex"),
        }

        # Process skin survey
        if data.get("skinSurvey"):
            for item in data["skinSurvey"]:
                field_id = item["id"]
                field_type = item["type"]
                field_value = item["value"]

                # Handle list types
                if field_type == "list":
                    try:
                        value_list = ast.literal_eval(field_value)
                        # Store the list values with index suffixes
                        for i, val in enumerate(value_list):
                            record[f"{field_id}_{i}"] = val
                        # Store the count
                        record[f"{field_id}_count"] = len(value_list)
                    except:
                        # Fallback if list parsing fails
                        record[field_id] = field_value
                else:
                    # Store non-list values directly
                    record[field_id] = field_value

        records.append(record)

    # Convert to DataFrame
    flat_df = pd.DataFrame(records)

    # Add back original columns
    result = pd.concat([
        df[["patientid", "customerid", "dateofbirth", "sex", "skinissuetype", "state"]].reset_index(drop=True),
        flat_df,
    ], axis=1)

    # Fill NaN values
    result = result.fillna("")

    return result


df = pd.read_csv("./patients-export.csv")
print("read CSV")
flattened_df = flatten_medical_background(df)
print("flattened")
flattened_df.to_csv("flattened_patients_explort.csv", index=False)
print("done")


# Example usage:
# df = pd.read_csv('your_input.csv', sep='\t')
# flattened_df = flatten_medical_background(df)
# flattened_df.to_csv('flattened_output.csv', index=False)

# To see what types of fields exist in your data:
def analyze_field_types(df):
    """
    Analyze all field types in the medical background data
    """
    field_types = set()

    for _, row in df.iterrows():
        if pd.isna(row["medicalbackground"]) or not row["medicalbackground"]:
            continue

        data = json.loads(row["medicalbackground"])
        if data.get("skinSurvey"):
            for item in data["skinSurvey"]:
                field_types.add((item["id"], item["type"]))

    return pd.DataFrame(list(field_types), columns=["field_id", "type"]).sort_values("type")
