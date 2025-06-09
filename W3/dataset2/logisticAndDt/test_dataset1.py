import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

#  Load dữ liệu test từ dataset 1
X_test = pd.read_csv("W2/X_test.csv")
y_test = pd.read_csv("W2/y_test.csv")["Severity Encoded"]

#  Load mô hình & scaler đã train từ dataset 2
logistic_model = joblib.load("W3/dataset2/logisticAndDt/logistic_model.pkl")
dt_model = joblib.load("W3/dataset2/logisticAndDt/decision_tree_model.pkl")
scaler = joblib.load("W3/dataset2/logisticAndDt/scaler.pkl")
train_columns = joblib.load("W3/dataset2/logisticAndDt/train_columns.pkl")

#  Nếu CVSS Score có, kiểm tra và chuẩn hóa về thang điểm 10
if "CVSS Score" in X_test.columns:
    X_test["CVSS Score"] = pd.to_numeric(X_test["CVSS Score"], errors="coerce")
    if X_test["CVSS Score"].max() <= 1.0:
        X_test["CVSS Score"] = X_test["CVSS Score"] * 10

#  Đồng bộ cột với dataset2 (các cột thiếu sẽ được điền giá trị 0)
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
