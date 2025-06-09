import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu đã mã hóa
file_path = "W2/encoded_cve_data.csv"
cve_data = pd.read_csv(file_path)

# Chọn các cột đầu vào (X) và nhãn (y)
X = cve_data[['CVSS Score', 'Severity_LOW', 'Severity_MEDIUM', 'Severity_UNKNOWN']]
y = cve_data['Severity Encoded']

# Chia tập train-test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xuất dữ liệu ra file CSV
X_train.to_csv("W2/X_train.csv", index=False)
X_test.to_csv("W2/X_test.csv", index=False)
y_train.to_csv("W2/y_train.csv", index=False)
y_test.to_csv("W2/y_test.csv", index=False)

print("Chia dữ liệu thành công! Đã lưu các file X_train.csv, X_test.csv, y_train.csv, y_test.csv")