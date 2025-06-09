import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file123 = pd.read_csv('W3/dataset2/merged.csv')

# thiết lập giao diện mặc định
sns.set_theme()

# ẩn thông báo không cần thiết, tránh làm nhiễu
import warnings
warnings.simplefilter('ignore')

top_vend = file123['vendor'].value_counts().head(n = 50)
top = list(top_vend.index)

top_vend.plot(kind = 'bar')
plt.title("50 công ty bị ảnh hưởng thường xuyên nhất")
plt.ylabel('Số lượng CVE')
plt.xticks(rotation = 90, size = 8)
plt.show()

file123_top = file123[file123['vendor'].isin(top)]
sns.barplot(x = 'vendor', y = 'cvss', data = file123_top, order = top)
plt.title("Điểm CVSS trung bình trên 50 công ty bị ảnh hưởng thường xuyên nhất")
plt.xticks(rotation = 90, size = 8)
plt.show()