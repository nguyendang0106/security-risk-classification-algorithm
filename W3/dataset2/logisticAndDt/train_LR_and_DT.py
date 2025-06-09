import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#  Load dataset1
data = pd.read_parquet("W3/dataset2/file123_train.parquet") 

#  Kiểm tra danh sách cột
print(" Danh sách cột trong DataFrame:", data.columns.tolist())

#  Kiểm tra cột không phải số
non_numeric_columns = data.select_dtypes(exclude=["number"]).columns.tolist()
print("🔍 Các cột không phải số:", non_numeric_columns)

#  Chuyển 'cvss' về số (nếu cần)
data["cvss"] = pd.to_numeric(data["cvss"], errors="coerce")

#  Nếu 'cvss' nằm trong khoảng 0-1, nhân 10 để đưa về thang điểm 10
if data["cvss"].max() <= 1.0:
    data["cvss"] = data["cvss"] * 10

#  Phân nhóm 'cvss' thành 'cvss_category'
bins = [-1, 0.1, 3.9, 6.9, 10]
labels = [0, 1, 2, 3]  # UNKNOWN - 0; LOW - 1; MEDIUM - 2; HIGH - 3
data["cvss_category"] = pd.cut(data["cvss"], bins=bins, labels=labels)

#  Loại bỏ các cột không cần thiết
drop_columns = ["cve_id", "mod_date", "pub_date", "summary", "cwe_code", "cwe_name", "cvss"]
X = data.drop(columns=drop_columns + ["cvss_category"])
y = data["cvss_category"].astype(int)

#  Mã hóa cột 'vulnerable_product' nếu có
if "vulnerable_product" in X.columns:
    encoder = LabelEncoder()
    X["vulnerable_product"] = encoder.fit_transform(X["vulnerable_product"])
    joblib.dump(encoder, "vulnerable_product_encoder.pkl")  # Lưu encoder

#  Chia train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

#  Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

#  Đánh giá mô hình
y_pred_lr = logistic_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f" Logistic Regression Accuracy: {accuracy_lr:.4f}")
print(f" Decision Tree Accuracy: {accuracy_dt:.4f}")

#  Lưu mô hình
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns, "train_columns.pkl")  # Lưu danh sách cột

print(" Mô hình đã được lưu thành công!")
