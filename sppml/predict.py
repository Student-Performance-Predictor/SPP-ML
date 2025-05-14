import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Define expected features for consistency
EXPECTED_FEATURES = [
    "Attendance_Percentage", "Homework_Completion_Percentage", "Parental_Education",
    "Study_Hours_Per_Week", "Failures", "Extra_Curricular", "Participation_Score",
    "Teacher_Rating", "Discipline_Issues", "Late_Submissions",
    "Previous_Grade_1", "Previous_Grade_2"
]

def load_model(model_path):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at: {full_path}")
    return joblib.load(full_path)

def load_scaler(scaler_path):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, scaler_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Scaler file not found at: {full_path}")
    return joblib.load(full_path)

def validate_input(data: pd.DataFrame):
    missing = [f for f in EXPECTED_FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return data[EXPECTED_FEATURES]

def predict_single(student_data: dict, model_path="models/lr_model.pkl", scaler_path="models/scaler.pkl"):
    try:
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)
        
        df = pd.DataFrame([student_data])
        df = validate_input(df)
        
        # Scale and wrap back in DataFrame
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

        prediction = model.predict(df_scaled)[0]
        return int(round(prediction))
    
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

def predict_bulk(input_data, model_path="models/lr_model.pkl", scaler_path="models/scaler.pkl", from_csv=False):
    try:
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)

        df = pd.read_csv(input_data) if from_csv else input_data.copy()
        df = validate_input(df)

        # Scale and wrap back in DataFrame
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

        predictions = model.predict(df_scaled)
        result_df = df.copy()
        result_df["Predicted_Final_Grade"] = [int(round(p)) for p in predictions]
        return result_df
    except Exception as e:
        print(f"Bulk prediction failed: {e}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    sample_input = {
        "Attendance_Percentage": 10,
        "Homework_Completion_Percentage": 20,
        "Parental_Education": 4,
        "Study_Hours_Per_Week": 10,
        "Failures": 5,
        "Extra_Curricular": 1,
        "Participation_Score": 2,
        "Teacher_Rating": 1,
        "Discipline_Issues": 5,
        "Late_Submissions": 8,
        "Previous_Grade_1": 20,
        "Previous_Grade_2": 10,
    }
    value = min(predict_single(sample_input),95)
    print(value)