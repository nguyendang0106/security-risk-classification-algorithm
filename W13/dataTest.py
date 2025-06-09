import pandas as pd

# Đọc tệp Parquet
df = pd.read_parquet('W9/data1/test2/combined_mapped_data/combined_mapped_sampled_data.parquet')

df.drop(columns=['Label', 'Timestamp', 'Group'], inplace=True, errors='ignore') # Xóa cột không cần thiết nếu có

# Lưu sang tệp CSV
# index=False để không ghi chỉ số của DataFrame vào tệp CSV
# header=False nếu tệp CSV của bạn không nên có dòng tiêu đề (tùy thuộc vào cách bạn muốn trang web xử lý)
# Nếu API của bạn chỉ cần các dòng dữ liệu thô, bạn có thể muốn header=False
df.to_csv('combined_mapped_sampled_data.csv', index=False, header=False) # Hoặc header=False

print("Đã chuyển đổi tệp Parquet sang CSV!")