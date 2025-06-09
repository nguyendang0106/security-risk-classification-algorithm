import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **1. Đọc dữ liệu test**
test_file = "W3/dataset2/file123_test.parquet"
test_data = pd.read_parquet(test_file)

# **2. Loại bỏ các cột không phải số**
test_data = test_data.select_dtypes(include=[np.number])

# **3. Tách feature (X) và label (y)**
X_test = test_data.drop(columns=["cvss"])  # Bỏ cột mục tiêu
y_test = test_data["cvss"]

# **4. Load scaler đã lưu**
scaler = joblib.load("W3/dataset2/scaler.pkl")
X_test_scaled = scaler.transform(X_test)  # Áp dụng scaler lên tập test

# **5. Load mô hình đã huấn luyện**
lr_model = joblib.load("W3/dataset2/linear_regression_model.pkl")
dt_model = joblib.load("W3/dataset2/decision_tree_model.pkl")

# **6. Dự đoán**
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test_scaled)

# **7. Đánh giá mô hình**
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nKết quả đánh giá mô hình {model_name}:")
    print(f"   - MAE  (Mean Absolute Error)  : {mae:.4f}")
    print(f"   - MSE  (Mean Squared Error)   : {mse:.4f}")
    print(f"   - RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print(f"   - R² Score                   : {r2:.4f}")

# **8. Hiển thị kết quả**
evaluate_model(y_test, y_pred_lr, "Hồi quy tuyến tính")
evaluate_model(y_test, y_pred_dt, "Decision Tree")

print("\n🎯 Kiểm tra hoàn tất!")
