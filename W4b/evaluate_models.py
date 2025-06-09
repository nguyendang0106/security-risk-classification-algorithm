import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import classification_report

# **Định nghĩa đường dẫn file**
DATA_PATH = "W4b"
X_TEST_PATH = f"{DATA_PATH}/X_test.npz"
Y_TEST_PATH = f"{DATA_PATH}/y_test.npy"
MODEL_PATHS = {
    "Logistic Regression": f"{DATA_PATH}/models/model_logistic_regression.pkl",
    "Random Forest": f"{DATA_PATH}/models/model_random_forest.pkl",
    "XGBoost": f"{DATA_PATH}/models/model_xgboost.pkl"
}

# **Tải dữ liệu test**
print(" Đang tải dữ liệu kiểm tra...")
X_test = load_npz(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f" X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# **Đánh giá từng mô hình**
for model_name, model_path in MODEL_PATHS.items():
    print(f"\n Đang đánh giá: {model_name}...")

    # **Tải mô hình**
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # **Dự đoán**
    y_pred = model.predict(X_test)

    # **In kết quả đánh giá**
    print(f"\n Kết quả đánh giá {model_name}:")
    print(classification_report(y_test, y_pred))

print("\n Hoàn thành đánh giá mô hình!")
