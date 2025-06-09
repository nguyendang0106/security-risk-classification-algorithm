from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import pandas as pd

# Đọc file JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

# Trích xuất văn bản mô tả và nhãn
X = []
y = []

for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    descriptions = cve.get("descriptions", [])
    
    # Lấy mô tả bằng tiếng Anh
    desc_text = ""
    for desc in descriptions:
        if desc["lang"] == "en":
            desc_text = desc["value"]
            break
    
    metrics = cve.get("metrics", {})
    
    # Lấy nhãn baseSeverity
    severity_label = None
    if "cvssMetricV2" in metrics:
        for metric in metrics["cvssMetricV2"]:
            if "baseSeverity" in metric:
                severity_label = metric["baseSeverity"]
                break

    if desc_text and severity_label:
        X.append(desc_text)
        y.append(severity_label)

# Chuyển văn bản thành đặc trưng TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Giới hạn số đặc trưng để giảm độ phức tạp
X_tfidf = vectorizer.fit_transform(X)

# Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Kiểm tra lại số lượng mẫu sau SMOTE
print("Số lượng mẫu sau khi cân bằng:")
print(pd.Series(y_train_resampled).value_counts())


# Số lượng mẫu sau khi cân bằng:
# HIGH      799
# MEDIUM    799
# LOW       799
# Name: count, dtype: int64
