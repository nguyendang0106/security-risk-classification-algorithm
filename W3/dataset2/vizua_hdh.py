import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file123 = pd.read_csv('W3/dataset2/merged.csv')

# thiết lập giao diện mặc định
sns.set_theme()

# ẩn thông báo không cần thiết, tránh làm nhiễu
import warnings
warnings.simplefilter('ignore')

# Trong biểu đồ thanh sản phẩm ở trên, ta có thể thấy rất nhiều "linux" hoặc "windows" trong tên
# xem liệu chúng ta có thể biết thêm thông tin về hệ điều hành từ tên sản phẩm không

file123['is_linux'] = np.where(file123['vulnerable_product'].str.lower().str.contains('linux'), 1, 0)
file123['is_android'] = np.where(file123['vulnerable_product'].str.lower().str.contains('android'), 2, 0)
file123['is_unix'] = np.where(file123['vulnerable_product'].str.lower().str.contains('unix'), 3, 0)
file123['is_chrome'] = np.where(file123['vulnerable_product'].str.lower().str.contains('chrome_os'), 4, 0)
file123['is_windows'] = np.where(file123['vulnerable_product'].str.lower().str.contains('windows'), 5, 0)
file123['is_mac'] = np.where(file123['vulnerable_product'].str.lower().str.contains('mac_os'), 6, 0)
file123['is_iphone'] = np.where(file123['vulnerable_product'].str.lower().str.contains('iphone_os'), 7, 0)

file123['os'] = file123['is_iphone'] + file123['is_linux'] + file123['is_mac'] + file123['is_android'] + file123['is_chrome'] + file123['is_windows'] + file123['is_unix']

# lập biểu đồ:

labels_by_order = ['linux', 'windows', 'mac_os', 'android', 'iphone_os', 'chrome_os', 'unix', 'unix & windows']
labels_by_order_with_non_os = ['non-OS / Other', 'linux', 'windows', 'mac_os', 'android', 'iphone_os', 'chrome_os', 'unix', 'unix & windows']
OS_count = file123['os'].value_counts()[1:]

OS_count.plot(kind = 'bar')
plt.title("CVE theo hệ điều hành")
plt.ylabel('Số lượng CVE')
plt.xticks(rotation = 90, size = 8, ticks = [0, 1, 2, 3, 4, 5, 6, 7], labels = labels_by_order)
plt.show()

sns.barplot(x = 'os', y = 'cvss', data = file123)
plt.title("Điểm CVSS trung bình trên các danh mục OS")
plt.xticks(rotation = 90, size = 8, ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8], labels = labels_by_order_with_non_os)
plt.show()

# Có lẽ điều này có liên quan đến bản chất mã nguồn mở của Linux khiến CVE của nó được báo cáo thường xuyên/dễ dàng hơn? Linux có nhiều CVE hơn đáng kể trong tập dữ liệu này nhưng điểm CVSS trung bình thấp hơn đáng kể so với các hệ điều hành phổ biến khác.

# Lưu ý: không phải tất cả các hệ điều hành đều được ghi lại trong biểu đồ này và có khả năng các thuật ngữ tìm kiếm của tôi cũng có thể loại trừ một số (ví dụ: 'mac_os' là một thuật ngữ tìm kiếm có thể bỏ sót một số hệ điều hành Mac. Nó đã giảm số lượng cve trong tìm kiếm đó xuống khoảng 1200 sự kiện, so với chỉ tìm kiếm 'mac'. Tuy nhiên, tôi đã chọn thuật ngữ độc quyền hơn để xóa các chương trình, như 'maconomy' có thể hiển thị trong kết quả của tôi với tìm kiếm ít nghiêm ngặt hơn)