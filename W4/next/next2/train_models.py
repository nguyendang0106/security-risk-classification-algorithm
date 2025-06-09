import json
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from scipy.sparse import save_npz
from imblearn.under_sampling import TomekLinks



# Đọc dữ liệu từ file JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

# Trích xuất mô tả và nhãn
X, y = [], []
for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    descriptions = cve.get("descriptions", [])
    
    desc_text = next((desc["value"] for desc in descriptions if desc["lang"] == "en"), "")
    
    metrics = cve.get("metrics", {})
    severity_label = next(
        (metric["baseSeverity"] for metric in metrics.get("cvssMetricV2", []) if "baseSeverity" in metric), None
    )

    if desc_text and severity_label:
        X.append(desc_text)
        y.append(severity_label)

# Chuyển văn bản thành vector TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  # Giữ nguyên giới hạn số feature
    ngram_range=(1,2),  # Thêm n-grams (unigram + bigram)
    stop_words="english",  # Loại bỏ stopwords
    sublinear_tf=True  # Giúp giảm ảnh hưởng của từ xuất hiện quá nhiều
)
X_tfidf = vectorizer.fit_transform(X)

# Chia dữ liệu train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Sau khi tách dữ liệu train/test, lưu lại tập test
save_npz("W4/next/next2/X_test.npz", X_test)
with open("W4/next/next2/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Giảm bớt dữ liệu dư thừa bằng TomekLinks
tomek = TomekLinks()
X_train_resampled, y_train_resampled = tomek.fit_resample(X_train_resampled, y_train_resampled)

# Lưu dữ liệu huấn luyện để dùng cho GridSearch
print(" Đang lưu dữ liệu huấn luyện...")
save_npz("W4/next/next2/X_train.npz", X_train_resampled)

# Đảm bảo nhãn đã được mã hóa số
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_resampled)

# Lưu dữ liệu huấn luyện đã được mã hóa
np.save("W4/next/next2/y_train.npy", y_train_encoded)
print(" Đã lưu X_train.npz và y_train.npy!")

# Mã hóa nhãn thành số (để dùng cho XGBoost)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_resampled)
y_test_encoded = label_encoder.transform(y_test)

# Lưu vectorizer & label encoder để sử dụng sau này
with open("W4/next/next2/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("W4/next/next2/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Khởi tạo danh sách mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# Huấn luyện & lưu mô hình
for model_name, model in models.items():
    print(f" Đang huấn luyện: {model_name}...")
    
    if model_name == "XGBoost":
        model.fit(X_train_resampled, y_train_encoded)  # Dùng nhãn số
    else:
        model.fit(X_train_resampled, y_train_resampled)  # Dùng nhãn gốc

    # Lưu mô hình đã huấn luyện
    # model_path = f"W4/next/next2/model_{model_name.lower().replace(' ', '_')}.pkl"
    # with open(model_path, "wb") as f:
    #     pickle.dump(model, f)
    
    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    
    if model_name == "XGBoost":
        y_pred = label_encoder.inverse_transform(y_pred)  # Chuyển số về nhãn
    
    print(f" Kết quả đánh giá {model_name}:")
    print(classification_report(y_test, y_pred))

print(" Hoàn thành huấn luyện và lưu mô hình!")



#  Đang huấn luyện: Logistic Regression...
#  Kết quả đánh giá Logistic Regression:
#               precision    recall  f1-score   support

#         HIGH       0.74      0.72      0.73       200
#          LOW       0.50      0.66      0.57        44
#       MEDIUM       0.68      0.63      0.66       150

#     accuracy                           0.68       394
#    macro avg       0.64      0.67      0.65       394
# weighted avg       0.69      0.68      0.68       394

#  Đang huấn luyện: Random Forest...
#  Kết quả đánh giá Random Forest:
#               precision    recall  f1-score   support

#         HIGH       0.70      0.78      0.73       200
#          LOW       0.61      0.52      0.56        44
#       MEDIUM       0.71      0.63      0.66       150

#     accuracy                           0.69       394
#    macro avg       0.67      0.64      0.65       394
# weighted avg       0.69      0.69      0.69       394

#  Đang huấn luyện: XGBoost...
#  Kết quả đánh giá XGBoost:
#               precision    recall  f1-score   support

#         HIGH       0.72      0.74      0.73       200
#          LOW       0.57      0.64      0.60        44
#       MEDIUM       0.71      0.66      0.69       150

#     accuracy                           0.70       394
#    macro avg       0.67      0.68      0.67       394
# weighted avg       0.70      0.70      0.70       394

#  Hoàn thành huấn luyện và lưu mô hình!


# 1.Accuracy (Độ chính xác):
# Mô hình XGBoost có độ chính xác cao nhất (0.70), tiếp theo là Random Forest (0.69) và Logistic Regression (0.68).

# 2.Precision, Recall, và F1-score:
# Logistic Regression có precision thấp nhất, đặc biệt trên lớp "LOW" (0.50). Điều này nghĩa là nó dễ dự đoán sai các mẫu thuộc lớp này.
# Random Forest có precision và recall khá cân bằng nhưng vẫn không vượt trội so với XGBoost.
# XGBoost có recall cao hơn ở lớp "LOW" (0.64 so với 0.66 của Logistic), giúp phát hiện đúng nhiều mẫu hơn trong lớp nhỏ này.

# 3.Hiệu suất theo từng lớp:
# "HIGH": Tất cả các mô hình đều có hiệu suất tốt.
# "LOW": XGBoost thể hiện tốt hơn Logistic Regression.
# "MEDIUM": Hiệu suất tương tự nhau giữa các mô hình.

# Kết luận
# XGBoost là mô hình tốt nhất dựa trên các chỉ số.