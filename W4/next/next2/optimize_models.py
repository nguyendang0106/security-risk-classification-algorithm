import os
import pickle
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Định nghĩa đường dẫn
X_TRAIN_PATH = "W4/next/next2/X_train.npz"
Y_TRAIN_PATH = "W4/next/next2/y_train.npy"
MODEL_SAVE_PATH = "W4/next/next2/models/"

# Tạo thư mục nếu chưa có
# os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load dữ liệu
print(" Đang tải dữ liệu huấn luyện...")
X_train = load_npz(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

# Chuyển nhãn thành số nếu cần
# if y_train.dtype.kind in {'U', 'O'}:  # Kiểm tra nếu y_train là chuỗi
#     le = LabelEncoder()
#     y_train = le.fit_transform(y_train)
#     joblib.dump(le, f"{MODEL_SAVE_PATH}label_encoder.pkl")  # Lưu LabelEncoder để dùng sau

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}, unique labels: {np.unique(y_train)}")

# ==============================
#  1. Tối ưu Logistic Regression (GridSearch)
# ==============================
log_reg_params = {
    "C": [0.001, 0.01, 0.1, 1, 10, 50],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

print("\n Đang tối ưu Logistic Regression...")
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, n_jobs=-1, verbose=1)
grid_log_reg.fit(X_train, y_train)

print(f" Best Logistic Regression Params: {grid_log_reg.best_params_}")
with open(f"{MODEL_SAVE_PATH}model_logistic_regression.pkl", "wb") as f:
    pickle.dump(grid_log_reg.best_estimator_, f)
# ==============================
#  2. Tối ưu Random Forest (RandomizedSearch)
# ==============================
rf_params = {
    "n_estimators": [50, 100, 200, 500, 1000],
    "max_depth": [5, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

print("\n Đang tối ưu Random Forest...")
random_rf = RandomizedSearchCV(RandomForestClassifier(), rf_params, n_iter=20, cv=5, n_jobs=-1, verbose=1, random_state=42)
random_rf.fit(X_train, y_train)

print(f" Best Random Forest Params: {random_rf.best_params_}")
with open(f"{MODEL_SAVE_PATH}model_random_forest.pkl", "wb") as f:
    pickle.dump(random_rf.best_estimator_, f)
# ==============================
#  3. Tối ưu XGBoost (RandomizedSearch)
# ==============================
xgb_params = {
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

print("\n Đang tối ưu XGBoost...")
random_xgb = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    xgb_params, n_iter=20, cv=5, n_jobs=-1, verbose=1, random_state=42
)
random_xgb.fit(X_train, y_train)

print(f" Best XGBoost Params: {random_xgb.best_params_}")
with open(f"{MODEL_SAVE_PATH}model_xgboost.pkl", "wb") as f:
    pickle.dump(random_xgb.best_estimator_, f)

print("\n Hoàn thành tối ưu hóa và lưu mô hình!")

