import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Đọc dữ liệu
X_train = pd.read_csv("W2/X_train.csv")
X_test = pd.read_csv("W2/X_test.csv")
y_train = pd.read_csv("W2/y_train.csv").values.ravel()  # Chuyển thành mảng 1D
y_test = pd.read_csv("W2/y_test.csv").values.ravel()

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Lưu mô hình để sử dụng sau này
joblib.dump(model, "logistic_regression_model_dataset1.pkl")

print("Mô hình đã được lưu thành công!")

# Accuracy: 1.0000
# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00         6
#            1       1.00      1.00      1.00        56
#            2       1.00      1.00      1.00       156
#            3       1.00      1.00      1.00       182

#     accuracy                           1.00       400
#    macro avg       1.00      1.00      1.00       400
# weighted avg       1.00      1.00      1.00       400

# Confusion Matrix:
#  [[  6   0   0   0]
#  [  0  56   0   0]
#  [  0   0 156   0]
#  [  0   0   0 182]]
# Mô hình đã được lưu thành công!

