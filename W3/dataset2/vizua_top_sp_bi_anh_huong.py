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

top_prod = file123['vulnerable_product'].value_counts().head(n = 50)
top_p = list(top_prod.index)

top_prod.plot(kind = 'bar')
plt.title("50 sản phẩm bị ảnh hưởng thường xuyên nhất")
plt.ylabel('Số lượng CVE')
plt.xticks(rotation = 90, size = 8)
plt.show()

file123_top_p = file123[file123['vulnerable_product'].isin(top_p)]
sns.barplot(x = 'vulnerable_product', y = 'cvss', data = file123_top_p, order = top_p)
plt.title("Điểm CVSS trung bình trên 50 sản phẩm bị ảnh hưởng thường xuyên nhất")
plt.xticks(rotation = 90, size = 8)
plt.show()