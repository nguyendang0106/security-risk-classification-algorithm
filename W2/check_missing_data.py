import pandas as pd

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

# Kiểm tra dữ liệu thiếu
missing_values = df.isnull().sum()
print("Số lượng giá trị thiếu trong mỗi cột:\n", missing_values)

# Số lượng giá trị thiếu trong mỗi cột:
#  CVE ID         0
# Description    0
# Severity       0
# CVSS Score     0
# References     0
# dtype: int64
