import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split_and_save(input_path, output_train_path, output_test_path, label_column):
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    # Chia dữ liệu theo tỉ lệ 50-50 và giữ nguyên tỉ lệ nhãn
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)
    
    # Lưu dữ liệu
    train_df.to_parquet(output_train_path, index=False)
    test_df.to_parquet(output_test_path, index=False)

# Đường dẫn và tên cột nhãn (giả sử cột nhãn tên là 'label', bạn sửa nếu khác)
label_col = 'Label'

# Benign
stratified_split_and_save(
    'W9/data2/clean/all_benign.parquet',
    'W9/data2/train/all_benign.parquet',
    'W9/data2/test/all_benign.parquet',
    label_col
)

# Malicious
stratified_split_and_save(
    'W9/data2/clean/all_malicious.parquet',
    'W9/data2/train/all_malicious.parquet',
    'W9/data2/test/all_malicious.parquet',
    label_col
)
