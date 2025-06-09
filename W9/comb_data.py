import pandas as pd

# Đọc hai file parquet
be_df1 = pd.read_parquet('W9/data1/train/all_benign.parquet')
be_df2 = pd.read_parquet('W9/data2/train/all_benign.parquet')

# Gộp hai dataframe lại
combined_df = pd.concat([be_df1, be_df2], ignore_index=True)

# Lưu ra file mới
combined_df.to_parquet('W9/dataComb/train/all_benign.parquet', index=False)



# Đọc hai file parquet
ma_df1 = pd.read_parquet('W9/data1/train/all_malicious.parquet')
ma_df2 = pd.read_parquet('W9/data2/train/all_malicious.parquet')

# Gộp hai dataframe lại
combined_df = pd.concat([ma_df1, ma_df2], ignore_index=True)

# Lưu ra file mới
combined_df.to_parquet('W9/dataComb/train/all_malicious.parquet', index=False)
