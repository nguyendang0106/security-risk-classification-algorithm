import pandas as pd

# Bước 1: Đọc dữ liệu
df = pd.read_csv("W2/preprocessed_cve_data.csv")

# Bước 2: Kiểm tra thông tin tổng quan
print(df.info())
print(df.describe())

# Bước 3: Đếm số lượng CVE theo mức độ Severity
severity_counts = df["Severity"].value_counts()
print("Số lượng CVE theo mức độ nguy hiểm:")
print(severity_counts)

# Bước 4: CVSS trung bình theo mức độ nguy hiểm
avg_cvss_by_severity = df.groupby("Severity")["CVSS Score"].mean()
print("Điểm CVSS trung bình theo mức độ nguy hiểm:")
print(avg_cvss_by_severity)

# Bước 5: Tìm CVE có CVSS cao nhất (nguy hiểm nhất)
top_cve = df[df["CVSS Score"] == df["CVSS Score"].max()]
print("CVE nguy hiểm nhất:")
print(top_cve)

# Bước 6: Nếu CVE ID có chứa năm (CVE-YYYY-XXXX), có thể phân tích xu hướng theo năm
df["Year"] = df["CVE ID"].apply(lambda x: int(x.split("-")[1]))  # Tách năm từ CVE ID
yearly_counts = df["Year"].value_counts().sort_index()
print("Số lượng CVE theo năm:")
print(yearly_counts)

# Xuất kết quả phân tích ra file
severity_counts.to_csv("W2/severity_distribution.csv")
avg_cvss_by_severity.to_csv("W2/cvss_avg_by_severity.csv")
yearly_counts.to_csv("W2/cve_counts_by_year.csv")

print("Phân tích hoàn tất! Kết quả đã được lưu vào file CSV.")


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2000 entries, 0 to 1999
# Data columns (total 5 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   CVE ID       2000 non-null   object 
#  1   Description  2000 non-null   object 
#  2   Severity     2000 non-null   object 
#  3   CVSS Score   2000 non-null   float64
#  4   References   2000 non-null   object 
# dtypes: float64(1), object(4)
# memory usage: 78.3+ KB
# None
#         CVSS Score
# count  2000.000000
# mean      6.252848
# std       2.246761
# min       0.000000
# 25%       5.000000
# 50%       6.600000
# 75%       7.500000
# max      10.000000
# Số lượng CVE theo mức độ nguy hiểm:
# Severity
# HIGH       999
# MEDIUM     747
# LOW        220
# UNKNOWN     34
# Name: count, dtype: int64
# Điểm CVSS trung bình theo mức độ nguy hiểm:
# Severity
# HIGH       8.100901
# LOW        2.084545
# MEDIUM     5.008969
# UNKNOWN    6.252848
# Name: CVSS Score, dtype: float64
# CVE nguy hiểm nhất:
#              CVE ID  ...                                         References
# 0     CVE-1999-0095  ...  http://www.osvdb.org/195, http://seclists.org/...
# 1     CVE-1999-0082  ...  http://www.alw.nih.gov/Security/Docs/admin-gui...
# 4     CVE-1999-1467  ...  https://exchange.xforce.ibmcloud.com/vulnerabi...
# 21    CVE-1999-1193  ...  http://www.securityfocus.com/bid/20, https://e...
# 26    CVE-1999-0498  ...  https://exchange.xforce.ibmcloud.com/vulnerabi...
# ...             ...  ...                                                ...
# 1959  CVE-2000-0405  ...  http://www.l0pht.com/advisories/asniff_advisor...
# 1967  CVE-2000-0437  ...  http://www.securityfocus.com/bid/1234, http://...
# 1978  CVE-2000-0551  ...  http://archives.neohapsis.com/archives/bugtraq...
# 1980  CVE-2000-0398  ...  http://www.securityfocus.com/bid/1244, http://...
# 1988  CVE-2000-0491  ...  http://archives.neohapsis.com/archives/bugtraq...

# [278 rows x 5 columns]
# Số lượng CVE theo năm:
# Year
# 1999    1538
# 2000     461
# 2001       1
# Name: count, dtype: int64
# Phân tích hoàn tất! Kết quả đã được lưu vào file CSV.


# Tổng quan về dữ liệu:
# Có tổng cộng 2000 CVE.
# Điểm CVSS trung bình: 6.25 (mức trung bình khá cao, có nhiều lỗ hổng nghiêm trọng).
# Điểm CVSS cao nhất: 10.0 (278 CVE thuộc nhóm nguy hiểm nhất).
# Mức độ nguy hiểm:
# HIGH: 999 CVE
# MEDIUM: 747 CVE
# LOW: 220 CVE
# UNKNOWN: 34 CVE

# Nhóm HIGH có điểm CVSS trung bình rất cao (~8.1), chứng tỏ đây là nhóm lỗ hổng cực kỳ nghiêm trọng.
# Nhóm LOW có điểm CVSS trung bình chỉ ~2.08, ít nguy hiểm hơn.

# Top CVE nguy hiểm nhất (CVSS = 10.0):
# Có 278 lỗ hổng nghiêm trọng nhất.
# Một số CVE nổi bật:
# CVE-1999-0095 (Sendmail Debug Command)
# CVE-1999-0082 (ftpd root access)
# CVE-1999-1467 (passwd Buffer Overflow)

# 1999 có số CVE cao nhất (1538), có thể do đây là giai đoạn bắt đầu công bố lỗ hổng an ninh hàng loạt.
# Số lượng CVE giảm mạnh vào năm 2000 và 2001 – có thể dữ liệu của bạn chỉ chứa CVE trong giai đoạn này.