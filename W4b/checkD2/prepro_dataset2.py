import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#  **Định nghĩa đường dẫn file**
DATASET_PATH = "W3/dataset2/merged.csv"  # Dữ liệu đầu vào
PROCESSED_DATA_PATH = "W4b/checkD2/preprocessed_dataset2.pkl"  # Dữ liệu đã tiền xử lý
LABEL_ENCODER_PATH = "W4b/checkD2/label_encoder.pkl"  # Encoder của CVSS category
TFIDF_VECTORIZER_PATH = "W4b/tfidf_vectorizer.pkl"  # Vectorizer của tập train
TRAIN_COLUMNS_PATH = "W4b/train_columns.pkl"  # Cấu trúc cột đã train

#  **1. Load dataset**
print(" Đang tải dataset...")
data = pd.read_csv(DATASET_PATH)

#  **2. Chuyển đổi CVSS thành nhãn**
print(" Chuyển đổi CVSS thành nhãn LOW / MEDIUM / HIGH...")
cvss_bins = [-1, 3.9, 6.9, 10]  # Ngưỡng phân loại
cvss_labels = ["LOW", "MEDIUM", "HIGH"]
data["cvss_category"] = pd.cut(data["cvss"], bins=cvss_bins, labels=cvss_labels)

#  **3. Mã hóa nhãn CVSS**
print(" Mã hóa nhãn CVSS...")
label_encoder = LabelEncoder()
data["cvss_encoded"] = label_encoder.fit_transform(data["cvss_category"])

#  **4. Chuyển đổi cột text (summary) thành vector TF-IDF**
print(" Chuyển đổi văn bản thành vector TF-IDF...")
with open(TFIDF_VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)  # Load vectorizer đã train trước

summary_tfidf = tfidf_vectorizer.transform(data["summary"].fillna(""))  # Xử lý dữ liệu thiếu

#  **5. One-hot encode các cột categorical**
print(" Mã hóa One-Hot Encoding các cột categorical...")
categorical_columns = [
    "access_authentication", "access_complexity", "access_vector",
    "impact_availability", "impact_confidentiality", "impact_integrity", "vendor"
]
 
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_encoded = one_hot_encoder.fit_transform(data[categorical_columns].fillna("UNKNOWN"))

#  **6. Kết hợp toàn bộ feature**
print(" Kết hợp tất cả các feature lại với nhau...")
X_numerical = data[["cvss"]].values  # Feature số học
X_combined = pd.concat([
    pd.DataFrame(X_numerical, columns=["cvss"]),
    pd.DataFrame(categorical_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
], axis=1)

#  **7. Đồng bộ feature với danh sách cột đã train**
print(" Đồng bộ danh sách cột với tập train...")
with open(TRAIN_COLUMNS_PATH, "rb") as f:
    train_columns = pickle.load(f)

X_combined = X_combined.reindex(columns=train_columns, fill_value=0)  # Thêm cột thiếu, loại bỏ cột thừa

#  **8. Lưu dataset đã xử lý**
print(" Lưu dataset đã xử lý...")
data_final = pd.concat([X_combined, pd.DataFrame(summary_tfidf.toarray())], axis=1)
data_final["cvss_encoded"] = data["cvss_encoded"]

with open(PROCESSED_DATA_PATH, "wb") as f:
    pickle.dump(data_final, f)

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print(" Tiền xử lý hoàn tất! Dataset đã sẵn sàng để đánh giá mô hình.")
