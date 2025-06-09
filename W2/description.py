import pandas as pd

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

df["Description Length"] = df["Description"].apply(len)
print(df["Description Length"].describe())

# count    2000.000000
# mean      141.726000
# std        66.981962
# min        20.000000
# 25%        95.000000
# 50%       129.000000
# 75%       172.000000
# max       705.000000
# Name: Description Length, dtype: float64