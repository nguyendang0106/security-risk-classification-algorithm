import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

plt.figure(figsize=(8, 5))
sns.boxplot(x="Severity", y="CVSS Score", data=df)
plt.title("Mối quan hệ giữa CVSS Score và Severity")
plt.show()