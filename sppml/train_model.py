import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_model(cleaned_file="data/students.csv", model_file="sppml/models/lr_model.pkl", scaler_file="sppml/models/scaler.pkl"):
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname("sppml/models"), exist_ok=True)

    df = pd.read_csv(cleaned_file)

    required_features = [
        "Attendance_Percentage", "Homework_Completion_Percentage", "Parental_Education",
        "Study_Hours_Per_Week", "Failures", "Extra_Curricular", "Participation_Score",
        "Teacher_Rating", "Discipline_Issues", "Late_Submissions",
        "Previous_Grade_1", "Previous_Grade_2"
    ]

    for feature in required_features + ["Final_Grade"]:
        if feature not in df.columns:
            raise ValueError(f"Missing required column: {feature}")

    df.fillna(df.mean(), inplace=True)

    X = df[required_features]
    y = df["Final_Grade"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to: {scaler_file}")

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc_5 = np.mean(np.abs(y_test - y_pred) <= 5)

    print(f"\n📈 Linear Regression Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy within ±5 marks: {acc_5 * 100:.2f}%")

    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")

    # Plot predictions
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Final Grades")
    plt.ylabel("Predicted Final Grades")
    plt.title("Actual vs Predicted Final Grades")
    plt.tight_layout()
    plt.savefig("sppml/models/actual_vs_predicted.png")
    print("Prediction plot saved to: sppml/models/actual_vs_predicted.png")


if __name__ == "__main__":
    train_model()
