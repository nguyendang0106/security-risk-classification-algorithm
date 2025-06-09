import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')

# Display the first few rows of the training set
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", train_df.columns)


print(train_df['label'].value_counts())         # 0: normal, 1: attack
print(train_df['attack_cat'].value_counts())    # attack category

# Check for missing values
print("Missing values in train set:")
train_df.info()
train_df.isnull().sum().sort_values(ascending=False)


# Check the number of unique values
print("Duplicates in train set:")
train_df['proto'].value_counts()
train_df['service'].value_counts()
train_df['state'].value_counts()


# Distribution of values ​​of some numerical features
print("Unique values in categorical columns:")
train_df[['dur', 'sbytes', 'dbytes', 'rate', 'sload', 'dload']].describe()


full_df = pd.read_csv('W6/data/UNSW-NB15_1.csv')
print(full_df.shape)
print(full_df.columns)


# Analysis between numerical features and labels
corr = train_df.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr[['label']].sort_values(by='label', ascending=False), annot=True, cmap='coolwarm')
plt.title('Tương quan giữa đặc trưng và label')
plt.show()

# Ratio of each attack_cat type to label = 1
attack_counts = train_df[train_df['label'] == 1]['attack_cat'].value_counts()
print(attack_counts)


# Không có missing values trong training set.
# Không có duplicate entries (giả sử bạn đã kiểm tra như train_df.duplicated().sum() == 0).
# Điểm cộng lớn: Dữ liệu đã khá sạch → bạn có thể đi thẳng vào phân tích, xử lý feature và model.

# 45 cột bao gồm:
# 30 numeric - int64
# 11 numeric - float64
# 4 categorical - object: proto, service, state, attack_cat
# label: Nhị phân (0 = bình thường, 1 = tấn công)
# attack_cat: Dạng phân loại nhiều lớp (10 loại tấn công + 1 loại "Normal")

print("=========================================================")


# Chỉ chọn các cột số (loại float và int)
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).drop(['label'], axis=1).columns

# Tính tương quan Pearson với nhãn
correlations = train_df[numeric_cols].corrwith(train_df['label'])

# Lấy các đặc trưng có tương quan mạnh (>|0.1|)
strong_corr = correlations[correlations.abs() > 0.1].sort_values(key=abs, ascending=False)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.barplot(x=strong_corr.index, y=strong_corr.values)
plt.title('Tương quan giữa đặc trưng số và nhãn (label)')
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()