import pandas as pd
import os

def preprocess_data(input_file: str, output_file: str) -> None:
    # Load raw data
    df = pd.read_csv(input_file)

    # Drop unnecessary columns
    columns_to_drop = ["School", "Student_ID", "Name", "Class"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Convert all columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Ensure Final_Grade is integer
    if "Final_Grade" in df.columns:
        df["Final_Grade"] = df["Final_Grade"].round().astype(int)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    preprocess_data("data/raw_students.csv", "data/students.csv")
