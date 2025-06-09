import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root (two directories up from the script)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# 📂 **Định nghĩa đường dẫn file với đường dẫn tuyệt đối**
PREPROCESSED_DATASET_PATH = os.path.join(PROJECT_ROOT, "W4b", "checkD2", "preprocessed_dataset2.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "W4b", "checkD2", "label_encoder.pkl")
TRAIN_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "W4b", "train_columns.pkl")
MODEL_PATHS = {
    "Logistic Regression": os.path.join(PROJECT_ROOT, "W4b", "models", "model_logistic_regression.pkl"),
    "Random Forest": os.path.join(PROJECT_ROOT, "W4b", "models", "model_random_forest.pkl"),
    "XGBoost": os.path.join(PROJECT_ROOT, "W4b", "models", "model_xgboost.pkl")
}

# Đảm bảo đường dẫn đúng
print(f"📂 Đường dẫn dữ liệu: {PREPROCESSED_DATASET_PATH}")
print(f"📂 Đường dẫn label encoder: {LABEL_ENCODER_PATH}")
print(f"📂 Đường dẫn danh sách cột: {TRAIN_COLUMNS_PATH}")

with open(TRAIN_COLUMNS_PATH, "rb") as f:
    train_columns = pickle.load(f)

print(f"Số đặc trưng khi huấn luyện: {len(train_columns)}")