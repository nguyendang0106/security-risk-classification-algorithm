import pandas as pd

file_path = "W3/dataset2/file123_train.parquet"

# Đọc file
data = pd.read_parquet(file_path)

# Kiểm tra cột dữ liệu
print(" Các cột trong dataset mới:", data.columns)

# Xem vài dòng đầu
print(data.head())

import pickle

# Load danh sách cột đã train
with open("W4b/train_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

print(f"Số cột trong train_columns.pkl: {len(train_columns)}")
print("Một số cột đầu tiên:", train_columns[:10])


