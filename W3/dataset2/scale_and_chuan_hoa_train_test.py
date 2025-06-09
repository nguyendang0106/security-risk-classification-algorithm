import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Đọc dữ liệu
train_file = "W3/dataset2/file123_train.parquet"
data = pd.read_parquet(train_file)

data = data.select_dtypes(include=[np.number])  # Giữ lại các cột số

# Chia thành feature (X) và target (y)
X = data.drop(columns=["cvss"])  # Các đặc trưng đầu vào
y = data["cvss"]  # Nhãn đầu ra

# Chia tập train-test (cố định random_state để nhất quán)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Sử dụng scaler cũ

# Lưu scaler để sử dụng sau này
joblib.dump(scaler, "scaler.pkl")

# Lưu dữ liệu đã chuẩn hóa để đảm bảo train-test giống nhau
np.savez("train_test_data.npz", X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test)

print("Tiền xử lý hoàn tất, scaler đã được lưu!")
