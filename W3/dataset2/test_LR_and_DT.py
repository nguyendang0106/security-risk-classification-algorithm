import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tải scaler và mô hình
scaler = joblib.load("W3/dataset2/scaler.pkl")
lr_model = joblib.load("W3/dataset2/linear_regression_model.pkl")
dt_model = joblib.load("W3/dataset2/decision_tree_model.pkl")

# Tải lại dữ liệu test
data = np.load("W3/dataset2/train_test_data.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# Dự đoán với mô hình hồi quy tuyến tính
y_pred_lr = lr_model.predict(X_test)

# Dự đoán với mô hình cây quyết định
y_pred_dt = dt_model.predict(X_test)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n Kết quả đánh giá mô hình {model_name}:")
    print(f"   - MAE  (Mean Absolute Error)  : {mae:.4f}")
    print(f"   - MSE  (Mean Squared Error)   : {mse:.4f}")
    print(f"   - RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print(f"   - R² Score                   : {r2:.4f}")

# Đánh giá mô hình
evaluate_model(y_test, y_pred_lr, "Hồi quy tuyến tính")
evaluate_model(y_test, y_pred_dt, "Decision Tree")

print("\n Kiểm tra hoàn tất!")
