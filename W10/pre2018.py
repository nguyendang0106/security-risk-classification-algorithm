import pandas as pd

# Đường dẫn tới file đầu vào
input_path = "W9/data2/clean/all_data.parquet"

# Đường dẫn tới file đầu ra
output_path = "W10/data2/test_selected_features_2018.parquet"

# Các cột cần giữ lại
selected_columns = [
    'Label',
    'Flow Bytes/s',
    'Fwd Packet Length Mean',
    'Packet Length Mean',
    'Avg Packet Size',
    'Fwd Packet Length Max',
    'Bwd Packet Length Std',
    'Bwd Packets Length Total',
    'Init Bwd Win Bytes',
    'Avg Bwd Segment Size',
    'Bwd Packets/s',
    'Bwd Packet Length Mean',
    'Flow Packets/s',
    'Fwd Seg Size Min',
    'Packet Length Variance',
    'Subflow Bwd Bytes'
]

# Đọc dữ liệu
df = pd.read_parquet(input_path)

# Giữ lại các cột cần thiết
df_selected = df[selected_columns]

# Ghi ra file parquet mới
df_selected.to_parquet(output_path, index=False)

print(f"Đã lưu file chứa các nhãn được chọn tại: {output_path}")
