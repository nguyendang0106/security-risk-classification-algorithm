import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

# Kiểm tra dữ liệu trước khi mã hóa
print("Dữ liệu trước khi mã hóa:")
print(df.head())

# Mã hóa cột Severity bằng Label Encoding
severity_mapping = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
df["Severity Encoded"] = df["Severity"].map(severity_mapping)

# LabelEncoder nếu không muốn gán thủ công
# severity_encoder = LabelEncoder()
# df["Severity Encoded"] = severity_encoder.fit_transform(df["Severity"])

# One-Hot Encoding (nếu cần biến đổi cột Severity thành các cột nhị phân.)
df = pd.get_dummies(df, columns=["Severity"], drop_first=True)

# Kiểm tra dữ liệu sau khi mã hóa
print("Dữ liệu sau khi mã hóa:")
print(df.head())

# Lưu lại dữ liệu đã mã hóa
df.to_csv("W2/encoded_cve_data.csv", index=False)
print("Dữ liệu đã được mã hóa và lưu thành công!")

# Dữ liệu trước khi mã hóa:
#           CVE ID  ...                                         References
# 0  CVE-1999-0095  ...  http://www.osvdb.org/195, http://seclists.org/...
# 1  CVE-1999-0082  ...  http://www.alw.nih.gov/Security/Docs/admin-gui...
# 2  CVE-1999-1471  ...  http://www.securityfocus.com/bid/4, http://www...
# 3  CVE-1999-1122  ...  http://www.securityfocus.com/bid/3, https://ex...
# 4  CVE-1999-1467  ...  https://exchange.xforce.ibmcloud.com/vulnerabi...

# [5 rows x 5 columns]
# Dữ liệu sau khi mã hóa:
#           CVE ID                                        Description  ...  Severity_MEDIUM Severity_UNKNOWN
# 0  CVE-1999-0095  The debug command in Sendmail is enabled, allo...  ...            False            False
# 1  CVE-1999-0082      CWD ~root command in ftpd allows root access.  ...            False            False
# 2  CVE-1999-1471  Buffer overflow in passwd in BSD based operati...  ...            False            False
# 3  CVE-1999-1122  Vulnerability in restore in SunOS 4.0.3 and ea...  ...             True            False
# 4  CVE-1999-1467  Vulnerability in rcp on SunOS 4.0.x allows rem...  ...            False            False

# [5 rows x 8 columns]
# Dữ liệu đã được mã hóa và lưu thành công!


# Severity Encoded: Chuyển đổi Severity thành số (LOW : 1, MEDIUM : 2, HIGH : 3, UNKNOWN : 0).
# Severity_LOW, Severity_MEDIUM, Severity_UNKNOWN: One-Hot Encoding, giúp mô hình dễ xử lý hơn.