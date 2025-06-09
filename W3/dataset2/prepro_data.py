import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#  Đọc dữ liệu gốc 
file_path = "W3/dataset2/merged.csv"
data = pd.read_csv(file_path)

#  **Hiển thị thông tin trước khi xử lý**
print("Trước khi xử lý:", data.info())

#  **Xử lý cột "mod_date" (chuyển timedelta thành số giây)**
try:
    data['mod_date'] = pd.to_timedelta(data['mod_date']).dt.total_seconds()
except Exception as e:
    print("Lỗi chuyển đổi mod_date:", e)

#  **Chuẩn hóa dữ liệu số**
scaler = MinMaxScaler()
numerical_cols = ['cvss', 'mod_date']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

#  **Xử lý các cột phân loại (One-Hot Encoding)**
categorical_cols = ['access_authentication', 'access_complexity', 'access_vector', 
                    'impact_availability', 'impact_confidentiality', 'impact_integrity', 'vendor']

# Giữ lại **1000 giá trị phổ biến nhất** 
for col in categorical_cols:
    top_values = data[col].value_counts().index[:1000]  
    data[col] = data[col].apply(lambda x: x if x in top_values else "OTHER")

# Áp dụng One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Gộp lại với dữ liệu chính
data = data.drop(columns=categorical_cols).reset_index(drop=True)
data = pd.concat([data, encoded_df], axis=1)

#  **Tạo tập train & test**
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#  **Lưu file ở định dạng Parquet (giảm dung lượng)**
train_file = "file123_train.parquet"
test_file = "file123_test.parquet"

train_data.to_parquet(train_file, index=False)
test_data.to_parquet(test_file, index=False)

print(f"Xử lý hoàn tất! Dữ liệu đã lưu tại:\n   {train_file} (train)\n  📂 {test_file} (test)")
print("Kích thước tập train:", train_data.shape)
print("Kích thước tập test:", test_data.shape)
