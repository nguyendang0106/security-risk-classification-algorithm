import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

#  Load dữ liệu test
test_file = "W3/dataset2/file123_test.parquet"
data = pd.read_parquet(test_file)

#  Đảm bảo 'cvss' là số
data["cvss"] = pd.to_numeric(data["cvss"], errors="coerce")

#  Nếu 'cvss' nằm trong khoảng 0-1, nhân 10 để đưa về thang điểm 10
if data["cvss"].max() <= 1.0:
    data["cvss"] = data["cvss"] * 10

#  Phân nhóm 'cvss' thành 'cvss_category'
bins = [-1, 0.1, 3.9, 6.9, 10]
labels = [0, 1, 2, 3]  # UNKNOWN - 0; LOW - 1; MEDIUM - 2; HIGH - 3
data["cvss_category"] = pd.cut(data["cvss"], bins=bins, labels=labels)

#  Bỏ các cột không liên quan
drop_columns = ["cve_id", "mod_date", "pub_date", "summary", "cwe_code", "cwe_name", "cvss"]
X_test = data.drop(columns=drop_columns + ["cvss_category"])
y_test = data["cvss_category"].astype(int)

#  Load mô hình & scaler
logistic_model = joblib.load("W3/dataset2/logisticAndDt/logistic_model.pkl")
dt_model = joblib.load("W3/dataset2/logisticAndDt/decision_tree_model.pkl")
scaler = joblib.load("W3/dataset2/logisticAndDt/scaler.pkl")
train_columns = joblib.load("W3/dataset2/logisticAndDt/train_columns.pkl")

#  Xử lý cột 'vulnerable_product'
if "vulnerable_product" in X_test.columns:
    encoder = joblib.load("W3/dataset2/logisticAndDt/vulnerable_product_encoder.pkl")

    # Kiểm tra giá trị mới trong 'vulnerable_product'
    known_classes = set(encoder.classes_)
    X_test["vulnerable_product"] = X_test["vulnerable_product"].apply(
        lambda x: x if x in known_classes else "UNKNOWN"
    )

    # Cập nhật encoder nếu chưa có "UNKNOWN"
    if "UNKNOWN" not in known_classes:
        encoder.classes_ = np.array(list(encoder.classes_) + ["UNKNOWN"])  # Chuyển thành NumPy array

    # Mã hóa lại cột 'vulnerable_product'
    X_test["vulnerable_product"] = encoder.transform(X_test["vulnerable_product"])

#  Đồng bộ cột với dataset train
X_test = X_test.reindex(columns=train_columns, fill_value=0)

#  Chuẩn hóa dữ liệu test
X_test_scaled = scaler.transform(X_test)

#  Dự đoán với Logistic Regression
y_pred_lr = logistic_model.predict(X_test_scaled)
print("\n Logistic Regression:")
print(f" Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

#  Dự đoán với Decision Tree
y_pred_dt = dt_model.predict(X_test)
print("\n Decision Tree:")
print(f" Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))
