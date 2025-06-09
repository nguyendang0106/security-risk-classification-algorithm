import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("W1/cleaned_cve_data.csv")

# Kiểm tra thông tin dữ liệu
print(df.info())
print(df.head())

# Kiểm tra giá trị trống
print(df.isnull().sum())

# Kiểm tra dữ liệu trùng lặp
print(f"Số dòng trùng lặp: {df.duplicated().sum()}")


# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2000 entries, 0 to 1999
# Data columns (total 5 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   CVE ID       2000 non-null   object 
#  1   Description  2000 non-null   object 
#  2   Severity     2000 non-null   object 
#  3   CVSS Score   1966 non-null   float64
#  4   References   1966 non-null   object 
# dtypes: float64(1), object(4)
# memory usage: 78.3+ KB
# None
#           CVE ID  ...                                         References
# 0  CVE-1999-0095  ...  http://seclists.org/fulldisclosure/2019/Jun/16...
# 1  CVE-1999-0082  ...  http://www.alw.nih.gov/Security/Docs/admin-gui...
# 2  CVE-1999-1471  ...  http://www.cert.org/advisories/CA-1989-01.html...
# 3  CVE-1999-1122  ...  http://www.cert.org/advisories/CA-1989-02.html...
# 4  CVE-1999-1467  ...  http://www.cert.org/advisories/CA-1989-07.html...

# [5 rows x 5 columns]
# CVE ID          0
# Description     0
# Severity        0
# CVSS Score     34
# References     34
# dtype: int64
# Số dòng trùng lặp: 0