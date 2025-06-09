import pandas as pd
import numpy as np

# Định nghĩa đường dẫn file đầu vào
train_file = "W3/dataset2/file123_train.parquet"
debug_log = "debug_log.txt"
debug_csv = "cvss_debug.csv"

# Đọc dữ liệu
data = pd.read_parquet(train_file)

# Ghi danh sách cột ra file debug
with open(debug_log, "w", encoding="utf-8") as log_file:
    log_file.write(" Danh sách cột trong DataFrame:\n")
    log_file.write(str(data.columns.tolist()) + "\n\n")

# Kiểm tra xem 'cvss' có tồn tại không
if "cvss" not in data.columns:
    with open(debug_log, "a", encoding="utf-8") as log_file:
        log_file.write(" Lỗi: Cột 'cvss' không tồn tại trong dataset!\n")
    print(" Lỗi: Cột 'cvss' không tồn tại trong dataset!")
    exit()

# Kiểm tra kiểu dữ liệu của cvss
cvss_dtype = data["cvss"].dtype
with open(debug_log, "a", encoding="utf-8") as log_file:
    log_file.write(f" Kiểu dữ liệu của 'cvss': {cvss_dtype}\n")

# Nếu 'cvss' không phải là số, cố gắng chuyển về float
if not np.issubdtype(cvss_dtype, np.number):
    data["cvss"] = pd.to_numeric(data["cvss"], errors="coerce")
    with open(debug_log, "a", encoding="utf-8") as log_file:
        log_file.write("⚠️ 'cvss' không phải số, đã chuyển về kiểu float.\n")

# Kiểm tra giá trị nhỏ nhất, lớn nhất và thống kê cơ bản của 'cvss'
cvss_min = data["cvss"].min()
cvss_max = data["cvss"].max()
cvss_stats = data["cvss"].describe()

with open(debug_log, "a", encoding="utf-8") as log_file:
    log_file.write(f"\n Giá trị nhỏ nhất: {cvss_min}\n")
    log_file.write(f" Giá trị lớn nhất: {cvss_max}\n")
    log_file.write("\n Thống kê mô tả:\n")
    log_file.write(str(cvss_stats) + "\n\n")

# Nếu cvss nằm trong khoảng [0, 1], nhân 10 để chuyển về thang điểm 10
if cvss_max <= 1.0:
    data["cvss"] = data["cvss"] * 10
    with open(debug_log, "a", encoding="utf-8") as log_file:
        log_file.write(" 'cvss' đã được nhân 10 để đưa về thang điểm 10.\n")

# Phân nhóm 'cvss' thành 'cvss_category'
bins = [-1, 0.1, 3.9, 6.9, 10]
labels = [0, 1, 2, 3]  # UNKNOWN - 0; LOW - 1; MEDIUM - 2; HIGH - 3
data["cvss_category"] = pd.cut(data["cvss"], bins=bins, labels=labels)

# Kiểm tra xem cột 'cvss_category' có tồn tại không
if "cvss_category" not in data.columns:
    with open(debug_log, "a", encoding="utf-8") as log_file:
        log_file.write(" Lỗi: Cột 'cvss_category' không được tạo thành công!\n")
    print(" Lỗi: Cột 'cvss_category' không được tạo thành công!")
    exit()

# Kiểm tra phân bố dữ liệu theo nhóm
category_counts = data["cvss_category"].value_counts().sort_index()

with open(debug_log, "a", encoding="utf-8") as log_file:
    log_file.write("\n Phân bố dữ liệu theo nhóm:\n")
    log_file.write(str(category_counts) + "\n")

# Kiểm tra xem có giá trị NaN không
nan_count = data["cvss_category"].isna().sum()
with open(debug_log, "a", encoding="utf-8") as log_file:
    log_file.write(f"\n Số lượng NaN trong 'cvss_category': {nan_count}\n")

# Lưu một phần dữ liệu để kiểm tra
data_sample = data[["cvss", "cvss_category"]].head(50)
data_sample.to_csv(debug_csv, index=False)

print(" Debug hoàn tất! Xem log trong 'debug_log.txt' và dữ liệu mẫu trong 'cvss_debug.csv'.")
