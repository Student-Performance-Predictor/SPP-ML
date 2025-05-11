import joblib
import pandas as pd
import os

# Define expected features for consistency
EXPECTED_FEATURES = [
    "Attendance_Percentage", "Homework_Completion_Percentage", "Parental_Education",
    "Study_Hours_Per_Week", "Failures", "Extra_Curricular", "Participation_Score",
    "Teacher_Rating", "Discipline_Issues", "Late_Submissions",
    "Previous_Grade_1", "Previous_Grade_2"
]

def load_model(model_path="models/rf_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def validate_input(data: pd.DataFrame):
    missing = [f for f in EXPECTED_FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return data[EXPECTED_FEATURES]

def predict_single(student_data: dict, model_path="models/rf_model.pkl"):
    try:
        model = load_model(model_path)
        df = pd.DataFrame([student_data])
        df = validate_input(df)
        prediction = model.predict(df)[0]
        print(f"Predicted Final Grade: {int(round(prediction))}")
        return int(round(prediction))
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

def predict_bulk(input_data, model_path="models/rf_model.pkl", from_csv=False):
    """
    Predict final grades for multiple students.

    Args:
        input_data (str | pd.DataFrame): CSV file path or DataFrame.
        model_path (str): Trained model path.
        from_csv (bool): True if input_data is a CSV file path.

    Returns:
        pd.DataFrame: DataFrame with predictions added.
    """
    try:
        model = load_model(model_path)
        df = pd.read_csv(input_data) if from_csv else input_data.copy()
        df = validate_input(df)
        predictions = model.predict(df)
        result_df = df.copy()
        result_df["Predicted_Final_Grade"] = [int(round(p)) for p in predictions]
        return result_df
    except Exception as e:
        print(f"[⚠️] Bulk prediction failed: {e}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Single prediction
    sample_input = {
        "Attendance_Percentage": 95,
        "Homework_Completion_Percentage": 100,
        "Parental_Education": 4,
        "Study_Hours_Per_Week": 40,
        "Failures": 0,
        "Extra_Curricular": 0,
        "Participation_Score": 2,
        "Teacher_Rating": 5,
        "Discipline_Issues": 1,
        "Late_Submissions": 0,
        "Previous_Grade_1": 90,
        "Previous_Grade_2": 100
    }
    predict_single(sample_input)

    # Bulk prediction from CSV
    # Ensure the file has the required 12 features
    # results = predict_bulk("data/new_students.csv", from_csv=True)
    # print(results)
