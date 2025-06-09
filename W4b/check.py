import numpy as np

# Load dữ liệu
X_train = np.load("W4b/X_train.npz")["arr_0"]  # "arr_0" là key mặc định nếu file chỉ chứa một mảng
X_test = np.load("W4b/X_test.npz")["arr_0"]
y_train = np.load("W4b/y_train.npy")
y_test = np.load("W4b/y_test.npy")

# Kiểm tra số cột (đặc trưng)
print(f"📊 Số đặc trưng trong X_train: {X_train.shape[1]}")
print(f"📊 Số đặc trưng trong X_test: {X_test.shape[1]}")