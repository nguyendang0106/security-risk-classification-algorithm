import pandas as pd

# Đọc file
df1 = pd.read_parquet("W9/data1/clean/all_benign.parquet")
df2 = pd.read_parquet("W9/data2/clean/all_benign.parquet")
df3 = pd.read_parquet("W9/dataComb/train/all_benign.parquet")
df4 = pd.read_parquet("W9/dataComb/train/all_malicious.parquet")

def report_nan_ratio(df, file_name):
    total_rows = len(df)
    nan_per_col = df.isna().sum()
    nan_ratio_per_col = (nan_per_col / total_rows) * 100

    print(f"--- {file_name} ---")
    print("Tổng số dòng:", total_rows)
    print("Số lượng và tỷ lệ NaN theo cột:\n")
    print(pd.DataFrame({
        "NaN count": nan_per_col,
        "NaN ratio (%)": nan_ratio_per_col.round(2)
    }).sort_values("NaN ratio (%)", ascending=False))

    total_nan = nan_per_col.sum()
    total_values = df.size
    total_nan_ratio = (total_nan / total_values) * 100
    print(f"\nTổng số NaN: {total_nan} ({total_nan_ratio:.2f}% của toàn bộ bảng)\n")

# Kiểm tra 4 file
print("------------------------------------------------------------------------------------------------------------")
report_nan_ratio(df1, "W9/data1/clean/all_benign.parquet")
print("------------------------------------------------------------------------------------------------------------")
report_nan_ratio(df2, "W9/data2/clean/all_benign.parquet")
print("------------------------------------------------------------------------------------------------------------")
report_nan_ratio(df3, "W9/dataComb/train/all_benign.parquet")
print("------------------------------------------------------------------------------------------------------------")
report_nan_ratio(df4, "W9/dataComb/train/all_malicious.parquet")
print("------------------------------------------------------------------------------------------------------------")
