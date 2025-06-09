import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# Load JSON dataset
with open("W1/raw_cve_data.json", "r", encoding="utf-8") as file:
    data_json = json.load(file)

cve_list = data_json["vulnerabilities"]
df_json = pd.json_normalize(cve_list, sep="_")

# Load CSV dataset
df_csv = pd.read_csv("W3/dataset2/merged.csv")

# Chuẩn hóa tên cột để đồng bộ
df_json = df_json.rename(columns={
    "cve_id": "cve_id",
    "cve_published": "published",
    "cve_lastModified": "lastModified",
    "metrics_cvssMetricV2_0_cvssData_baseScore": "cvss",
    "metrics_cvssMetricV2_0_baseSeverity": "severity",
    "descriptions_0_value": "summary"
})

df_csv = df_csv.rename(columns={"cwe_code": "cwe_code", "summary": "summary", "cvss": "cvss", "severity": "severity"})

# Ghép hai dataset
df = pd.concat([df_json, df_csv], ignore_index=True)

# Xử lý dữ liệu
# Chuyển đổi thời gian thành số ngày kể từ 1970
df["published"] = pd.to_datetime(df["published"], errors="coerce")
df["lastModified"] = pd.to_datetime(df["lastModified"], errors="coerce")

df["days_since_published"] = (df["published"] - pd.Timestamp("1970-01-01")).dt.days
df["days_since_modified"] = (df["lastModified"] - pd.Timestamp("1970-01-01")).dt.days

# Điền NaN cho các cột thời gian
df["days_since_published"] = df["days_since_published"].fillna(df["days_since_published"].median())
df["days_since_modified"] = df["days_since_modified"].fillna(df["days_since_modified"].median())

# Điền giá trị thiếu
df["cvss"] = df["cvss"].fillna(df["cvss"].median())

# Hàm chuyển điểm CVSS thành mức độ nghiêm trọng
def classify_severity(cvss_score):
    if cvss_score < 4.0:
        return "LOW"
    elif cvss_score < 7.0:
        return "MEDIUM"
    else:
        return "HIGH"

df["severity"] = df["cvss"].apply(classify_severity)

# Mã hóa categorical features
categorical_cols = ["cwe_code"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Chuẩn hóa tất cả đặc trưng số
scaler = StandardScaler()
num_features = ["days_since_published", "days_since_modified", "cvss"]
df[num_features] = scaler.fit_transform(df[num_features])

# Xử lý văn bản với TF-IDF
vectorizer = TfidfVectorizer(max_features=500)  # Chọn 500 đặc trưng phổ biến nhất
df["summary"] = df["summary"].fillna("")  # Xử lý NaN
summary_tfidf = vectorizer.fit_transform(df["summary"])

# Chọn các đặc trưng để train
features = num_features + categorical_cols
X_numerical = df[features]
X = hstack([X_numerical, summary_tfidf])  # Kết hợp đặc trưng số và đặc trưng văn bản
y_reg = df["cvss"]  # Dự đoán điểm CVSS
y_class = df["severity"].fillna("MEDIUM")  # Dự đoán mức độ nghiêm trọng

# Mã hóa nhãn mức độ nghiêm trọng
severity_encoder = LabelEncoder()
y_class = severity_encoder.fit_transform(y_class)

#  **SMOTE cho phân loại (Classification)**
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled_class, y_resampled_class = smote.fit_resample(X, y_class)

# Chia train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_resampled_class, y_resampled_class, test_size=0.2, random_state=42
)

# Kiểm tra lại
print(df[features].isna().sum())  # Xác nhận không còn NaN trước khi train
print(df["severity"].isna().sum())  # Kiểm tra số lượng giá trị NaN trong cột severity
print(f"X shape: {X.shape}, y_reg shape: {y_reg.shape}, y_class shape: {y_class.shape}")
print(f"Missing values in X: {np.isnan(X.toarray()).sum()}")  # Kiểm tra NaN trong X (vì dùng hstack)

print(df["severity"].value_counts(normalize=True))  # Xem tỉ lệ phân bố nhãn
print(df["cvss"].describe())  # Kiểm tra range điểm CVSS
print(df[num_features].describe())  # Kiểm tra scale sau chuẩn hóa



# Train mô hình Random Forest cho regression (dự đoán điểm CVSS)
regressor = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"Mean Absolute Error (CVSS score after SMOTER): {mae}")
# Mean Absolute Error (CVSS score): 5.471760295653492e-06

# Train mô hình Random Forest cho classification (dự đoán mức độ nghiêm trọng)
classifier = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
classifier.fit(X_train_class, y_train_class)
y_pred_class = classifier.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy (Severity classification after SMOTE): {accuracy}")
# Accuracy (Severity classification): 0.9999385195507828

# # Đánh giá tổng quát bằng Cross-validation
# rf_scores = cross_val_score(classifier, X, y_class, cv=5, scoring="accuracy")
# print(f"Cross-validation accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")





# Train XGBoost cho Regression (Dự đoán CVSS Score)
xgb_regressor = XGBRegressor(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42)
xgb_regressor.fit(X_train_reg, y_train_reg)
y_pred_xgb_reg = xgb_regressor.predict(X_test_reg)
mae_xgb = mean_absolute_error(y_test_reg, y_pred_xgb_reg)
print(f"XGBoost Mean Absolute Error (CVSS score): {mae_xgb}")

# Train XGBoost cho Classification (Dự đoán mức độ nghiêm trọng)
xgb_classifier = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42)
xgb_classifier.fit(X_train_class, y_train_class)
y_pred_xgb_class = xgb_classifier.predict(X_test_class)
accuracy_xgb = accuracy_score(y_test_class, y_pred_xgb_class)
print(f"XGBoost Accuracy (Severity classification): {accuracy_xgb}")




# Train LightGBM cho Regression
lgbm_regressor = LGBMRegressor(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42)
lgbm_regressor.fit(X_train_reg, y_train_reg)
y_pred_lgbm_reg = lgbm_regressor.predict(X_test_reg)
mae_lgbm = mean_absolute_error(y_test_reg, y_pred_lgbm_reg)
print(f"LightGBM Mean Absolute Error (CVSS score): {mae_lgbm}")

# Train LightGBM cho Classification
lgbm_classifier = LGBMClassifier(n_estimators=50, max_depth=10, learning_rate=0.1, random_state=42)
lgbm_classifier.fit(X_train_class, y_train_class)
y_pred_lgbm_class = lgbm_classifier.predict(X_test_class)
accuracy_lgbm = accuracy_score(y_test_class, y_pred_lgbm_class)
print(f"LightGBM Accuracy (Severity classification): {accuracy_lgbm}")

import joblib

# Lưu mô hình Random Forest
joblib.dump(regressor, "random_forest_regressor.pkl")
joblib.dump(classifier, "random_forest_classifier.pkl")

# Lưu mô hình XGBoost
joblib.dump(xgb_regressor, "xgboost_regressor.pkl")
joblib.dump(xgb_classifier, "xgboost_classifier.pkl")

# Lưu mô hình LightGBM
joblib.dump(lgbm_regressor, "lightgbm_regressor.pkl")
joblib.dump(lgbm_classifier, "lightgbm_classifier.pkl")

# Lưu các bộ biến đổi dữ liệu để sử dụng lại khi predict
joblib.dump(scaler, "scaler.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(severity_encoder, "severity_encoder.pkl")

print("Mô hình và các bộ biến đổi đã được lưu thành công!")




from sklearn.model_selection import cross_val_score

# Hàm tính Cross-validation Score
def print_cross_val_score(model_name, model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")  # Dùng 5-Fold Cross Validation
    print(f"\n===== Cross-validation ({model_name}) =====")
    print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Cross-validation cho Random Forest
print_cross_val_score("Random Forest", classifier, X, y_class)

# Cross-validation cho XGBoost
print_cross_val_score("XGBoost", xgb_classifier, X, y_class)

# Cross-validation cho LightGBM
print_cross_val_score("LightGBM", lgbm_classifier, X, y_class)

# y_train_pred = regressor.predict(X_train)
# y_test_pred = regressor.predict(X_test)

# train_mae = mean_absolute_error(y_train_reg, y_train_pred)
# test_mae = mean_absolute_error(y_test_reg, y_test_pred)

# print(f"Train MAE: {train_mae}")
# print(f"Test MAE: {test_mae}")





# # Cross-validation

# cv_mae = cross_val_score(regressor, X, y_reg, cv=5, scoring="neg_mean_absolute_error")
# print(f"Cross-validation MAE: {-cv_mae.mean()}")






# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# cm = confusion_matrix(y_test_class, classifier.predict(X_test))  # Chạy với classifier, không phải regressor!
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=severity_encoder.classes_, yticklabels=severity_encoder.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


# Mean Absolute Error (CVSS score): 5.471760295653492e-06
# Accuracy (Severity classification): 0.9999385195507828
# Train MAE: 2.82299207254049e-06
# Test MAE: 5.471760295653492e-06
# Cross-validation MAE: 0.0004061158074192166
# 2025-03-28 10:11:08.097 Python[44957:8169952] +[IMKClient subclass]: chose IMKClient_Modern
# 2025-03-28 10:11:08.097 Python[44957:8169952] +[IMKInputSession subclass]: chose IMKInputSession_Modern


# 1️ Confusion Matrix
# Nhìn vào confusion matrix:

# HIGH (17,935 dự đoán đúng, không có lỗi).

# LOW (3,932 đúng, nhưng có 3 lần bị dự đoán nhầm thành MEDIUM).

# MEDIUM (26,926 đúng, không có lỗi).

#  Nhận xét:

# Mô hình phân loại hoạt động rất chính xác. Chỉ có 3 lỗi nhỏ khi dự đoán LOW nhưng bị nhầm thành MEDIUM.

# Điều này phù hợp với độ chính xác Accuracy = 99.99%, tức là hầu hết các điểm dữ liệu được phân loại đúng.

# 2️ Mean Absolute Error (MAE)
# Train MAE: 2.82e-06

# Test MAE: 5.47e-06

# Cross-validation MAE: 0.000406

#  Nhận xét:

# MAE rất nhỏ, gần như bằng 0, chứng tỏ mô hình hầu như không mắc lỗi khi dự đoán CVSS score.

# Tuy nhiên, MAE trong cross-validation cao hơn một chút (0.000406), có thể do một số tập dữ liệu khác nhau khi cross-validate.

#  Tóm lại, mô hình của bạn có tốt không?
# ✔ Phân loại (severity): Cực kỳ chính xác (gần như hoàn hảo).
# ✔ Hồi quy (CVSS score): Dự đoán cực kỳ sát với giá trị thực (MAE gần như bằng 0).
# ⚠ Có thể kiểm tra thêm:

# Biểu đồ phân phối lỗi (residuals plot) để chắc chắn mô hình không bị overfitting.

# Kiểm tra dữ liệu mới để xem mô hình có thực sự tổng quát hay không.


# # Residual Plot (Mục tiêu: Residuals nên phân bố đều quanh 0, không có mẫu lệch rõ ràng)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Tính toán residuals (chênh lệch giữa giá trị thực và giá trị dự đoán)
# y_pred = regressor.predict(X_test_reg)  # Dự đoán giá trị
# residuals = y_test_reg - y_pred

# # Vẽ đồ thị residuals
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals, bins=50, kde=True)
# plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
# plt.title("Residual Plot")
# plt.xlabel("Residual (Sai số)")
# plt.ylabel("Tần suất")
# plt.show()



# # Feature Importance (Mục tiêu: Xác định xem mô hình đang dựa vào những feature nào để dự đoán.)
# import pandas as pd

# # Lấy độ quan trọng của feature từ mô hình (nếu là RandomForest hoặc XGBoost)
# feature_importance = regressor.feature_importances_

# # Sắp xếp và hiển thị top 10 feature quan trọng nhất
# feature_names = list(X_numerical.columns) + list(vectorizer.get_feature_names_out())

# print(f"Length of feature_names: {len(feature_names)}")
# print(f"Length of feature_importance: {len(feature_importance)}")
# min_len = min(len(feature_names), len(feature_importance))
# importance_df = pd.DataFrame({
#     'Feature': feature_names[:min_len],
#     'Importance': feature_importance[:min_len]
# })
# importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

# plt.figure(figsize=(12, 6))
# sns.barplot(data=importance_df, x='Importance', y='Feature')
# plt.title("Feature Importance")
# plt.xlabel("Tầm quan trọng")
# plt.ylabel("Feature")
# plt.show()



#  các mẫu bị dự đoán sai trong confusion matrix (Mục tiêu: Xem có nhóm nào bị dự đoán sai nhiều không (đặc biệt là giữa HIGH, MEDIUM, LOW).)
from sklearn.metrics import confusion_matrix, classification_report

# Hàm hiển thị báo cáo cho từng mô hình
def print_classification_report(model_name, y_true, y_pred):
    print(f"\n===== {model_name} =====")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Random Forest
y_pred_rf = classifier.predict(X_test_class)
print_classification_report("Random Forest", y_test_class, y_pred_rf)

# XGBoost
y_pred_xgb = xgb_classifier.predict(X_test_class)
print_classification_report("XGBoost", y_test_class, y_pred_xgb)

# LightGBM
y_pred_lgbm = lgbm_classifier.predict(X_test_class)
print_classification_report("LightGBM", y_test_class, y_pred_lgbm)







import matplotlib.pyplot as plt
plt.hist(y_reg, bins=20)
plt.xlabel("CVSS Score")
plt.ylabel("Frequency")
plt.title("Distribution of Resampled CVSS Scores")
plt.show()

plt.hist(y_resampled_class, bins=3)
plt.xlabel("Severity Level")
plt.ylabel("Frequency")
plt.title("Distribution of Resampled Severity")
plt.xticks(ticks=[0,1,2], labels=severity_encoder.classes_)
plt.show()


# - **Precision cao (~97-99%)**: Mô hình rất ít dự đoán sai khi phân loại.
# - **Recall cũng cao (~96-99%)**: Mô hình bắt được hầu hết các mẫu thực tế.
# - **F1-score từ 0.97 đến 0.98**: Mô hình đạt hiệu suất cao ở tất cả các lớp.

# ---

# ### **Kết luận**
# - **Dữ liệu sau SMOTE/SMOTER đã được cân bằng tốt**, giúp mô hình tránh bị thiên lệch.
# - **Mô hình phân loại đạt accuracy 97.7% và có F1-score rất cao (~0.98)** → Hiệu suất rất tốt.
# - **Dự đoán CVSS score có MAE cực kỳ thấp (~0.0000034)** → Mô hình dự đoán rất chính xác.
# - **Một số lỗi nhỏ trong việc nhầm lẫn giữa HIGH và MEDIUM**, có thể cải thiện bằng cách tinh chỉnh feature selection hoặc threshold.

# Tóm lại, kết quả hiện tại là **rất tốt** và có thể áp dụng để đánh giá rủi ro định lượng trong bảo mật DevOps!