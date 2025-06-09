import pandas as pd

# Đọc file CSV
df = pd.read_csv("W3/dataset2/merged.csv")

# Kiểm tra khoảng giá trị CVSS
min_cvss = df["cvss"].min()
max_cvss = df["cvss"].max()

print(f"Giá trị CVSS nằm trong khoảng: {min_cvss} - {max_cvss}")

# Giá trị CVSS nằm trong khoảng: 1.2 - 10.0