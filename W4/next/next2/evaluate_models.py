import pickle
import numpy as np
from sklearn.metrics import classification_report
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder

# Định nghĩa đường dẫn file
TFIDF_PATH = "W4/next/next2/tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "W4/next/next2/label_encoder.pkl"
X_TEST_PATH = "W4/next/next2/X_test.npz"
Y_TEST_PATH = "W4/next/next2/y_test.pkl"
MODEL_PATHS = {
    "Logistic Regression": "W4/next/next2/models/model_logistic_regression.pkl",
    "Random Forest": "W4/next/next2/models/model_random_forest.pkl",
    "XGBoost": "W4/next/next2/models/model_xgboost.pkl"
}

# Load TF-IDF vectorizer (nếu cần dùng để transform dữ liệu)
with open(TFIDF_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load LabelEncoder để chuyển đổi nhãn y_test
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load tập test
X_test = load_npz(X_TEST_PATH)
with open(Y_TEST_PATH, "rb") as f:
    y_test = pickle.load(f)

# Chuyển đổi y_test từ dạng text sang số (đồng bộ với tập train)
y_test_encoded = label_encoder.transform(y_test)

# Đánh giá từng mô hình
for model_name, model_path in MODEL_PATHS.items():
    print(f"\n Đang đánh giá: {model_name}...")

    # Load mô hình
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Chuyển đổi y_pred sang số nếu cần
    if isinstance(y_pred[0], str):
        y_pred = label_encoder.transform(y_pred)

    # In kết quả đánh giá
    print(f"Kết quả đánh giá {model_name}:")
    print(classification_report(y_test_encoded, y_pred))

print("\n Đánh giá hoàn thành!")


# Đang đánh giá: Logistic Regression...
# Kết quả đánh giá Logistic Regression:
#               precision    recall  f1-score   support

#            0       0.74      0.72      0.73       200
#            1       0.50      0.66      0.57        44
#            2       0.68      0.63      0.66       150

#     accuracy                           0.68       394
#    macro avg       0.64      0.67      0.65       394
# weighted avg       0.69      0.68      0.68       394


#  Đang đánh giá: Random Forest...
# Kết quả đánh giá Random Forest:
#               precision    recall  f1-score   support

#            0       0.70      0.78      0.73       200
#            1       0.61      0.52      0.56        44
#            2       0.71      0.63      0.66       150

#     accuracy                           0.69       394
#    macro avg       0.67      0.64      0.65       394
# weighted avg       0.69      0.69      0.69       394


#  Đang đánh giá: XGBoost...
# Kết quả đánh giá XGBoost:
#               precision    recall  f1-score   support

#            0       0.72      0.74      0.73       200
#            1       0.57      0.64      0.60        44
#            2       0.71      0.66      0.69       150

#     accuracy                           0.70       394
#    macro avg       0.67      0.68      0.67       394
# weighted avg       0.70      0.70      0.70       394


#  Đánh giá hoàn thành!
