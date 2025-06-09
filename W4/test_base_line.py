import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

#  Load mô hình và vectorizer
with open("W4/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("W4/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

#  Load tập test đã lưu
test_df = pd.read_csv("W4/test_data.csv")

X_test = test_df["description"]
y_test = test_df["severity_label"]

#  Chuyển đổi văn bản test thành vector TF-IDF
X_test_tfidf = vectorizer.transform(X_test)

#  Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f" Độ chính xác của mô hình trên tập test: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["LOW", "MEDIUM", "HIGH"]))



#  Độ chính xác của mô hình trên tập test: 0.6827
#               precision    recall  f1-score   support

#          LOW       0.71      0.21      0.32        48
#       MEDIUM       0.68      0.65      0.66       155
#         HIGH       0.69      0.83      0.75       191

#     accuracy                           0.68       394
#    macro avg       0.69      0.56      0.58       394
# weighted avg       0.69      0.68      0.66       394


# "LOW" có vấn đề

# Precision cao (0.71) nhưng Recall quá thấp (0.21) → Mô hình ít khi dự đoán LOW, nhưng khi dự đoán thì đúng
# Điều này có thể do tập dữ liệu mất cân bằng, hoặc TF-IDF không thể phân biệt rõ class này.
# "HIGH" hoạt động tốt nhất

# Precision = 0.69, Recall = 0.83, F1 = 0.75 → Mô hình dự đoán class này khá ổn.



# Đánh giá tổng quan
# Tốt: Mô hình có khả năng nhận diện "HIGH" tốt nhất (F1-score = 0.75).
# Chưa tốt: Class "LOW" có Recall rất thấp (0.21) → Dễ bỏ sót các lỗ hổng nghiêm trọng thuộc nhóm này.



#  => Cần kiểm tra cân bằng dữu liệu xem sao