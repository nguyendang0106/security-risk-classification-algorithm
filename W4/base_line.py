import json
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

#  Load dữ liệu từ JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

#  Trích xuất thông tin cần thiết
cve_list = []
severity_list = []

for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    descriptions = cve.get("descriptions", [])
    metrics = cve.get("metrics", {})

    # Lấy mô tả bằng tiếng Anh (nếu có)
    description_text = ""
    for desc in descriptions:
        if desc["lang"] == "en":
            description_text = desc["value"]
            break

    # Lấy nhãn baseSeverity
    severity_label = None
    if "cvssMetricV2" in metrics:
        for metric in metrics["cvssMetricV2"]:
            if "baseSeverity" in metric:
                severity_label = metric["baseSeverity"]
                break
    
    # Chỉ thêm nếu có đầy đủ dữ liệu
    if description_text and severity_label:
        cve_list.append(description_text)
        severity_list.append(severity_label)

# Tạo DataFrame
df = pd.DataFrame({"description": cve_list, "severity": severity_list})

# Chuẩn hóa văn bản (xóa ký tự đặc biệt, chữ thường)
def clean_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Xóa ký tự đặc biệt
    return text

df["clean_description"] = df["description"].apply(clean_text)

# Chuyển đổi nhãn severity thành số
severity_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
df["severity_label"] = df["severity"].map(severity_mapping)

#  Chia tập train/test và lưu riêng tập test
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_description"], df["severity_label"], test_size=0.2, random_state=42
)

# Lưu tập test để sử dụng sau này
test_df = pd.DataFrame({"description": X_test, "severity_label": y_test})
test_df.to_csv("test_data.csv", index=False)

#  Trích xuất đặc trưng văn bản bằng TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

#  Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

#  Lưu mô hình và TF-IDF vectorizer
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(" Mô hình đã được huấn luyện và lưu thành công!")
print(" Dữ liệu test đã được lưu vào test_data.csv")
