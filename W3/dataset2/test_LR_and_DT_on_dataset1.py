import pandas as pd
import numpy as np
import joblib

# Load mô hình và scaler
lr_model = joblib.load("W3/dataset2/linear_regression_model.pkl")
dt_model = joblib.load("W3/dataset2/decision_tree_model.pkl")
scaler = joblib.load("W3/dataset2/scaler.pkl")

# Load tập test của dataset 1
X_test = pd.read_csv("W2/X_test.csv")
y_test = pd.read_csv("W2/y_test.csv")

# Lấy danh sách cột đã được sử dụng khi train
expected_columns = scaler.feature_names_in_

# Thêm các cột còn thiếu vào X_test với giá trị 0
for col in expected_columns:
    if col not in X_test.columns:
        X_test[col] = 0

# Đảm bảo X_test có đúng thứ tự cột như khi train
X_test = X_test[expected_columns]

# Chuẩn hóa dữ liệu
X_test_scaled = scaler.transform(X_test)

# Dự đoán với mô hình
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)

# Tính toán các chỉ số đánh giá
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n Kết quả đánh giá mô hình {model_name}:")
    print(f"   - MAE  (Mean Absolute Error)  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"   - MSE  (Mean Squared Error)   : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"   - RMSE (Root Mean Squared Error) : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"   - R² Score                   : {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_test, y_pred_lr, "Hồi quy tuyến tính")
evaluate_model(y_test, y_pred_dt, "Decision Tree")

print("\n Kiểm tra hoàn tất!")
