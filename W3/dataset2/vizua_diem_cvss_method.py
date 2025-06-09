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

cols = file123.columns[8:14]
file123['access_vector'] = file123['access_vector'].str.replace('_NETWORK', ' NET')

fig, ax = plt.subplots(nrows = 2, ncols = 3, sharey = True, layout = 'constrained', figsize = (8, 7))
i = 0
for axs in ax.flat:
    if i < 3:
        sns.barplot(x = cols[i], y = 'cvss', data = file123, ax = axs)
    else:
        sns.barplot(x = cols[i], y = 'cvss', data = file123, ax = axs, order = ['NONE', 'PARTIAL', 'COMPLETE'])
    axs.set_title('Điểm CVSS theo \n{}'.format(cols[i]), size = 10)
    i += 1
    axs.tick_params(labelrotation = 90, labelsize = 6)
plt.show()