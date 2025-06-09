import pandas as pd

# Đọc dữ liệu
train_file = "W3/dataset2/file123_train.parquet"
data = pd.read_parquet(train_file)

# Xem khoảng giá trị của cvss
print(" Giá trị nhỏ nhất:", data["cvss"].min())
print(" Giá trị lớn nhất:", data["cvss"].max())

# Xem phân bố giá trị cvss
print("\n Thống kê mô tả:")
print(data["cvss"].describe())

# Đếm số lượng giá trị theo khoảng
print("\n Phân bố dữ liệu theo khoảng:")
print(pd.cut(data["cvss"], bins=[0, 3.9, 6.9, 8.9, 10], labels=["UNKNOWN", "LOW", "MEDIUM", "HIGH"]).value_counts())
