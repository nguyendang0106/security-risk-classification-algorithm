import pandas as pd

# Đọc dữ liệu
train_file = "W3/dataset2/file123_train.parquet"
data = pd.read_parquet(train_file)

# Chuyển đổi cvss từ thang 0-1 sang 0-10
data["cvss"] = data["cvss"] * 10

# Phân nhóm giống dataset 1
data["cvss_category"] = pd.cut(
    data["cvss"], 
    bins=[0, 3.9, 6.9, 8.9, 10], 
    labels=[0, 1, 2, 3]  # 0: UNKNOWN, 1: LOW, 2: MEDIUM, 3: HIGH
)

# Kiểm tra lại phân bố
print("\n Phân bố dữ liệu sau khi chuyển đổi:")
print(data["cvss_category"].value_counts())

# Lưu lại dataset mới để dùng cho Logistic Regression
data.to_parquet("file123_train_converted.parquet")
print("\n Dữ liệu đã được chuyển đổi và lưu lại!")
