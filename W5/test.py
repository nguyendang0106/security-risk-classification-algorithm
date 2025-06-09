import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your data into df_combined
df_combined = pd.read_csv('your_data.csv')  # Replace with your actual data source

# 1️ Chuẩn hóa dữ liệu (Xử lý NaN)
df_combined.fillna("UNKNOWN", inplace=True)  

# 2️ Tách tập dữ liệu theo nguồn
df_json_data = df_combined[df_combined["cve_id"].str.startswith("CVE-199")]  # Giả sử CVE từ JSON có pattern này
df_csv_data = df_combined[~df_combined["cve_id"].str.startswith("CVE-199")]

# 3️ Chuyển đổi categorical features thành dạng số
df_encoded = pd.get_dummies(df_combined, columns=["access_vector", "access_complexity", "authentication"])

# 4️ Train/Test Split cho từng dataset
X_json = df_encoded.loc[df_json_data.index].drop(columns=["cve_id", "severity"])
y_json = df_json_data["severity"]

X_csv = df_encoded.loc[df_csv_data.index].drop(columns=["cve_id", "severity"])
y_csv = df_csv_data["severity"]

X_train_json, X_test_json, y_train_json, y_test_json = train_test_split(X_json, y_json, test_size=0.2, random_state=42)
X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(X_csv, y_csv, test_size=0.2, random_state=42)

# 5️ Train model riêng biệt
model_json = RandomForestClassifier(n_estimators=100, random_state=42)
model_json.fit(X_train_json, y_train_json)

model_csv = RandomForestClassifier(n_estimators=100, random_state=42)
model_csv.fit(X_train_csv, y_train_csv)

# 6️ Kiểm tra từng model trên dataset tương ứng
print(" Model JSON trên JSON test set:")
print(classification_report(y_test_json, model_json.predict(X_test_json)))

print(" Model CSV trên CSV test set:")
print(classification_report(y_test_csv, model_csv.predict(X_test_csv)))

# 7️ Cross-dataset Evaluation
print(" Model JSON trên CSV test set:")
print(classification_report(y_test_csv, model_json.predict(X_test_csv)))

print(" Model CSV trên JSON test set:")
print(classification_report(y_test_json, model_csv.predict(X_test_json)))
