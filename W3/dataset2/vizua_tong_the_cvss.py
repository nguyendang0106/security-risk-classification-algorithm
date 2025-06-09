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

sns.histplot(data = file123, x ='cvss', bins = 20)
plt.title("Phân phối tổng thể điểm CVSS trong Bộ dữ liệu")
plt.show()