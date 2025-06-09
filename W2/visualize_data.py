import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

# Biểu đồ số lượng CVE theo mức độ nguy hiểm
plt.figure(figsize=(8, 5))
sns.countplot(x="Severity", data=df, order=["LOW", "MEDIUM", "HIGH", "UNKNOWN"], palette="coolwarm")
plt.title("Số lượng CVE theo mức độ nguy hiểm")
plt.xlabel("Mức độ nguy hiểm")
plt.ylabel("Số lượng CVE")
# plt.savefig("so_luong_cve_theo_muc_nguy_hiem.png", dpi=300)
plt.show()

# Biểu đồ phân bố điểm CVSS
plt.figure(figsize=(8, 5))
sns.histplot(df["CVSS Score"], bins=20, kde=True, color="blue")
plt.title("Phân bố điểm CVSS của các CVE")
plt.xlabel("CVSS Score")
plt.ylabel("Số lượng CVE")
# plt.savefig("phan_bo_diem_CVSS.png", dpi=300)
plt.show()

# Biểu đồ số lượng CVE theo năm
df["Year"] = df["CVE ID"].str.extract(r'(\d{4})').astype(float)
year_counts = df["Year"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", color="red")
plt.title("Số lượng CVE theo năm")
plt.xlabel("Năm")
plt.ylabel("Số lượng CVE")
plt.xticks(rotation=45)
# plt.savefig("so_luong_CVE_theo_nam.png", dpi=300)
plt.show()
