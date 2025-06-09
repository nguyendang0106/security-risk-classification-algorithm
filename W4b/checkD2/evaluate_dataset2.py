# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.metrics import classification_report

# # 📂 **Định nghĩa đường dẫn file**
# PREPROCESSED_DATASET_PATH = "W4b/checkD2/preprocessed_dataset2.pkl"  # Dataset2 đã tiền xử lý
# LABEL_ENCODER_PATH = "W4b/checkD2/label_encoder.pkl"  # LabelEncoder đã train trước đó
# TRAIN_COLUMNS_PATH = "W4b/train_columns.pkl"  # Danh sách đặc trưng đã train
# MODEL_PATHS = {
#     "Logistic Regression": "W4b/models/model_logistic_regression.pkl",
#     "Random Forest": "W4b/models/model_random_forest.pkl",
#     "XGBoost": "W4b/models/model_xgboost.pkl"
# }

# # 📌 **1. Load dataset đã tiền xử lý**
# print("📂 Đang tải dataset mới...")
# with open(PREPROCESSED_DATASET_PATH, "rb") as f:
#     data = pickle.load(f)

# # **Tách tập test:**
# X_test = data.drop(columns=["cvss_encoded"])  # Loại bỏ nhãn
# y_test = data["cvss_encoded"].values  # Nhãn đã mã hóa

# print(f"📊 Dữ liệu test có {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng.")

# # 📌 **2. Load danh sách đặc trưng TF-IDF đã train**
# print("📂 Đang tải danh sách đặc trưng đã train...")
# with open(TRAIN_COLUMNS_PATH, "rb") as f:
#     train_columns = pickle.load(f)

# # 📌 **Đồng bộ tập test với danh sách cột đã train**
# missing_cols = set(train_columns) - set(X_test.columns)
# extra_cols = set(X_test.columns) - set(train_columns)

# print(f"📊 Đang kiểm tra tính nhất quán của tập test...")
# print(f"🛠️  Số cột thiếu trong tập test: {len(missing_cols)}")
# print(f"🛠️  Số cột dư trong tập test: {len(extra_cols)}")

# # **Thêm cột thiếu với giá trị 0 và loại bỏ cột dư**
# X_test = X_test.reindex(columns=train_columns, fill_value=0)

# print(f"📊 Số đặc trưng sau khi đồng bộ: {X_test.shape[1]} (Train: {len(train_columns)})")

# # 📌 **3. Load LabelEncoder để chuyển đổi nhãn**
# print("📂 Đang tải LabelEncoder...")
# with open(LABEL_ENCODER_PATH, "rb") as f:
#     label_encoder = pickle.load(f)

# # 📌 **4. Đánh giá từng mô hình**
# for model_name, model_path in MODEL_PATHS.items():
#     print(f"\n🚀 Đang đánh giá: {model_name}...")

#     # **Load mô hình**
#     with open(model_path, "rb") as f:
#         model = pickle.load(f)

#     # **Dự đoán**
#     y_pred = model.predict(X_test)

#     # **Chuyển đổi y_pred từ số sang nhãn**
#     y_pred_labels = label_encoder.inverse_transform(y_pred)
#     y_test_labels = label_encoder.inverse_transform(y_test)

#     # **In kết quả đánh giá**
#     print(f"\n📊 Kết quả đánh giá {model_name}:")
#     print(classification_report(y_test_labels, y_pred_labels))

# print("\n✅ Đánh giá hoàn thành!")


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

# 📌 **1. Load dataset đã tiền xử lý**
print("📂 Đang tải dataset mới...")
try:
    with open(PREPROCESSED_DATASET_PATH, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {PREPROCESSED_DATASET_PATH}")
    exit(1)

# **Tách tập test:**
X_test = data.drop(columns=["cvss_encoded"])  # Loại bỏ nhãn
y_test = data["cvss_encoded"].values  # Nhãn đã mã hóa

print(f"📊 Dữ liệu test có {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng.")

# 📌 **2. Load danh sách đặc trưng TF-IDF đã train**
print("📂 Đang tải danh sách đặc trưng đã train...")
try:
    with open(TRAIN_COLUMNS_PATH, "rb") as f:
        train_columns = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {TRAIN_COLUMNS_PATH}")
    exit(1)

# 📌 **Đồng bộ tập test với danh sách cột đã train**
print(f"📊 Số cột trong dữ liệu test: {len(X_test.columns)}")
print(f"📊 Số cột trong dữ liệu train: {len(train_columns)}")

missing_cols = set(train_columns) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(train_columns)

print(f"📊 Đang kiểm tra tính nhất quán của tập test...")
print(f"🛠️  Số cột thiếu trong tập test: {len(missing_cols)}")
if len(missing_cols) > 0:
    print(f"🛠️  Ví dụ về một số cột thiếu: {list(missing_cols)[:5]}")
print(f"🛠️  Số cột dư trong tập test: {len(extra_cols)}")
if len(extra_cols) > 0:
    print(f"🛠️  Ví dụ về một số cột dư: {list(extra_cols)[:5]}")

# **Tạo DataFrame mới với các cột giống tập train**
X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=train_columns)

# **Sao chép dữ liệu từ X_test sang X_test_aligned cho các cột tồn tại trong cả hai**
common_cols = set(X_test.columns).intersection(set(train_columns))
for col in common_cols:
    X_test_aligned[col] = X_test[col]

print(f"📊 Số đặc trưng sau khi đồng bộ: {X_test_aligned.shape[1]} (Train: {len(train_columns)})")

# 📌 **3. Load LabelEncoder để chuyển đổi nhãn**
print("📂 Đang tải LabelEncoder...")
try:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {LABEL_ENCODER_PATH}")
    exit(1)

# 📌 **4. Đánh giá từng mô hình**
for model_name, model_path in MODEL_PATHS.items():
    print(f"\n🚀 Đang đánh giá: {model_name}...")

    # **Load mô hình**
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy mô hình {model_path}")
        continue

    # **Dự đoán**
    try:
        y_pred = model.predict(X_test_aligned)
        
        # **Chuyển đổi y_pred từ số sang nhãn**
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)

        # **In kết quả đánh giá**
        print(f"\n📊 Kết quả đánh giá {model_name}:")
        print(classification_report(y_test_labels, y_pred_labels))
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán với mô hình {model_name}: {str(e)}")

print("\n✅ Đánh giá hoàn thành!")