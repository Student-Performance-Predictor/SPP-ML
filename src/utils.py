from pathlib import Path

# Base directory (root of your ML repo)
BASE_DIR = Path(__file__).parent.parent  # Adjust if needed

# Data directories
DATA_DIR = BASE_DIR / "data"            # Folder for all data files
MODEL_DIR = BASE_DIR / "model"          # Folder for saved models/scalers

# File paths
RAW_DATA_PATH = DATA_DIR / "raw_students.csv"          # Input raw data
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"  # Output cleaned data
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "model.joblib"

# Encoders
SCHOOL_ENCODER_PATH = "model/enc_school.joblib"
CLASS_ENCODER_PATH = "model/enc_class.joblib"
SECTION_ENCODER_PATH = "model/enc_section.joblib"
PARENT_EDU_ENCODER_PATH = "model/enc_parent_edu.joblib"
