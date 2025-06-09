import pandas as pd
import numpy as np

# Đọc dữ liệu
df_json = pd.read_json("W1/raw_cve_data.json")  
df_csv = pd.read_csv("W3/dataset2/merged.csv")  

print("Columns in df_json:", df_json.columns)
print("Columns in df_csv:", df_csv.columns)

print("JSON dataset size:", df_json.shape)
print("CSV dataset size:", df_csv.shape)

# Điền giá trị thiếu: thêm cột accessVector_ADJACENT_NETWORK vào JSON (mặc định = 0)
df_json["accessVector_ADJACENT_NETWORK"] = 0  

# Hợp nhất hai dataset
df_combined = pd.concat([df_json, df_csv], ignore_index=True)

# Chuẩn hóa phân phối severity (undersampling hoặc oversampling)
min_count = df_combined["severity"].value_counts().min()
df_balanced = df_combined.groupby("severity").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Kiểm tra lại dữ liệu
print(df_balanced["severity"].value_counts())  # Xem phân phối có đều không
