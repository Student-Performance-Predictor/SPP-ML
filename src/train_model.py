import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from preprocess import preprocess_data

def train_model(cleaned_file="data/students.csv", model_file="models/rf_model.pkl"):
    # Load cleaned data
    df = pd.read_csv(cleaned_file)

    # Split features and target
    X = df.drop(columns=["Final_Grade"])
    y = df["Final_Grade"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Trained")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")

if __name__ == "__main__":
    train_model()
