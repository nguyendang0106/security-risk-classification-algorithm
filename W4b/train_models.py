import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# **Định nghĩa đường dẫn file**
DATA_PATH = "W4b"
X_TRAIN_PATH = f"{DATA_PATH}/X_train.npz"
Y_TRAIN_PATH = f"{DATA_PATH}/y_train.npy"
X_TEST_PATH = f"{DATA_PATH}/X_test.npz"
Y_TEST_PATH = f"{DATA_PATH}/y_test.npy"
MODEL_SAVE_PATH = f"{DATA_PATH}/models/"

# **Tạo thư mục nếu chưa có**
# import os
# os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# **Tải dữ liệu**
print(" Đang tải dữ liệu huấn luyện...")
X_train = load_npz(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_test = load_npz(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f" X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f" X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# **Danh sách mô hình cần huấn luyện**
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# **Huấn luyện, lưu mô hình và đánh giá**
for model_name, model in models.items():
    print(f"\n Đang huấn luyện: {model_name}...")
    model.fit(X_train, y_train)

    # **Lưu mô hình**
    # model_path = f"{MODEL_SAVE_PATH}model_{model_name.lower().replace(' ', '_')}.pkl"
    # with open(model_path, "wb") as f:
    #     pickle.dump(model, f)

    # **Dự đoán**
    y_pred = model.predict(X_test)

    # **Đánh giá mô hình**
    print(f"\n Kết quả đánh giá {model_name}:")
    print(classification_report(y_test, y_pred))

print("\n Hoàn thành huấn luyện và lưu mô hình!")

