import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dữ liệu đã được xử lý
file_path = "W2/encoded_cve_data.csv" 
df = pd.read_csv(file_path)

# Tách features (X) và labels (y)
X = df.drop(columns=["CVE ID", "Description", "References", "Severity Encoded"])  # Loại bỏ cột không cần thiết
y = df["Severity Encoded"]

# Chia train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Khởi tạo và huấn luyện mô hình Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Lưu mô hình
joblib.dump(model, "decision_tree_model_dataset1.pkl")
print("Mô hình đã được lưu thành công!")

# Accuracy: 1.0000
# Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         6
#            1       1.00      1.00      1.00        56
#            2       1.00      1.00      1.00       156
#            3       1.00      1.00      1.00       182

#     accuracy                           1.00       400
#    macro avg       1.00      1.00      1.00       400
# weighted avg       1.00      1.00      1.00       400

# Confusion Matrix:
# [[  6   0   0   0]
#  [  0  56   0   0]
#  [  0   0 156   0]
#  [  0   0   0 182]]
# Mô hình đã được lưu thành công!
