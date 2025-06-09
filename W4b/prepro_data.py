import json
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from scipy.sparse import save_npz, hstack
from imblearn.under_sampling import TomekLinks


# Đọc file JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

# **Trích xuất dữ liệu**
X_text, X_numeric, X_categorical, y = [], [], [], []

for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    descriptions = cve.get("descriptions", [])
    metrics = cve.get("metrics", {})
    
    # **Mô tả bằng tiếng Anh**
    desc_text = next((desc["value"] for desc in descriptions if desc["lang"] == "en"), "")
    
    # **Nhãn baseSeverity**
    severity_label = next(
        (metric["baseSeverity"] for metric in metrics.get("cvssMetricV2", []) if "baseSeverity" in metric), None
    )
    
    # **Các đặc trưng số**
    base_score = next((metric["cvssData"]["baseScore"] for metric in metrics.get("cvssMetricV2", []) if "cvssData" in metric), None)
    exploitability = next((metric["exploitabilityScore"] for metric in metrics.get("cvssMetricV2", []) if "exploitabilityScore" in metric), None)
    impact = next((metric["impactScore"] for metric in metrics.get("cvssMetricV2", []) if "impactScore" in metric), None)

    # **Các đặc trưng phân loại**
    access_vector = next((metric["cvssData"]["accessVector"] for metric in metrics.get("cvssMetricV2", []) if "cvssData" in metric), None)
    access_complexity = next((metric["cvssData"]["accessComplexity"] for metric in metrics.get("cvssMetricV2", []) if "cvssData" in metric), None)
    authentication = next((metric["cvssData"]["authentication"] for metric in metrics.get("cvssMetricV2", []) if "cvssData" in metric), None)

    # **Lưu dữ liệu nếu đầy đủ**
    if desc_text and severity_label:
        X_text.append(desc_text)
        X_numeric.append([base_score, exploitability, impact])
        X_categorical.append([access_vector, access_complexity, authentication])
        y.append(severity_label)

# **Chuyển danh sách thành DataFrame**
df_numeric = pd.DataFrame(X_numeric, columns=["baseScore", "exploitabilityScore", "impactScore"])
df_categorical = pd.DataFrame(X_categorical, columns=["accessVector", "accessComplexity", "authentication"])

# **Xử lý dữ liệu**
# Xử lý văn bản với TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  # Giữ nguyên giới hạn số feature
    ngram_range=(1,2),  # Thêm n-grams (unigram + bigram)
    stop_words="english",  # Loại bỏ stopwords
    sublinear_tf=True  # Giúp giảm ảnh hưởng của từ xuất hiện quá nhiều
)
X_text_tfidf = vectorizer.fit_transform(X_text)

# Mã hóa dữ liệu phân loại
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_categorical_encoded = encoder.fit_transform(df_categorical)

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(df_numeric)

# **Ghép tất cả các đặc trưng lại**
X_final = hstack([X_text_tfidf, X_numeric_scaled, X_categorical_encoded])

# **Mã hóa nhãn**
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# **Chia tập train-test**
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded
)

# **Cân bằng dữ liệu với SMOTE**
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Giảm bớt dữ liệu dư thừa bằng TomekLinks
tomek = TomekLinks()
X_train_resampled, y_train_resampled = tomek.fit_resample(X_train_resampled, y_train_resampled)

# **Lưu dữ liệu**
save_npz("W4b/X_train.npz", X_train_resampled)
np.save("W4b/y_train.npy", y_train_resampled)
save_npz("W4b/X_test.npz", X_test)
np.save("W4b/y_test.npy", y_test)

# Lưu danh sách đặc trưng TF-IDF đã train
train_columns = vectorizer.get_feature_names_out().tolist()
TRAIN_COLUMNS_PATH = "W4b/train_columns.pkl"

with open(TRAIN_COLUMNS_PATH, "wb") as f:
    pickle.dump(train_columns, f)

print(f" Đã lưu danh sách đặc trưng TF-IDF vào {TRAIN_COLUMNS_PATH}")

# Lưu vectorizer, scaler, encoder
with open("W4b/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("W4b/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("W4b/onehot_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("W4b/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(" Dữ liệu đã tiền xử lý và lưu thành công!")
