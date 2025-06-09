import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ===== Đọc Dữ Liệu JSON (Dataset 1) =====
def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy danh sách CVE từ key "vulnerabilities"
    cve_list = data["vulnerabilities"]

    records = []
    for item in cve_list:
        cve = item["cve"]
        cve_id = cve["id"]
        base_severity = "UNKNOWN"

        # Lấy mức độ rủi ro (baseSeverity)
        if "metrics" in cve and "cvssMetricV2" in cve["metrics"]:
            severity_info = cve["metrics"]["cvssMetricV2"][0]
            base_severity = severity_info["baseSeverity"]

        # Lấy mô tả bằng tiếng Anh
        description = ""
        if "descriptions" in cve:
            for desc in cve["descriptions"]:
                if desc["lang"] == "en":
                    description = desc["value"]
                    break

        # Lấy các đặc trưng khác
        access_vector = None
        if "metrics" in cve and "cvssMetricV2" in cve["metrics"]:
            access_vector = cve["metrics"]["cvssMetricV2"][0]["cvssData"]["accessVector"]

        records.append([cve_id, description, access_vector, base_severity])

    return pd.DataFrame(records, columns=["cve_id", "description", "accessVector", "severity"])


# ===== Đọc Dữ Liệu CSV (Dataset 2) =====
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)

    # Chuyển đổi CVSS thành nhãn (LOW, MEDIUM, HIGH)
    def map_cvss_to_severity(cvss):
        if cvss < 4.0:
            return "LOW"
        elif 4.0 <= cvss < 7.0:
            return "MEDIUM"
        else:
            return "HIGH"

    df["severity"] = df["cvss"].apply(map_cvss_to_severity)
    
    # Lấy các cột cần thiết
    df = df[["cve_id", "summary", "access_vector", "severity"]]
    df.rename(columns={"summary": "description", "access_vector": "accessVector"}, inplace=True)

    return df


# ===== Tiền xử lý văn bản =====
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r"\W+", " ", text)  # Loại bỏ ký tự đặc biệt
    text = " ".join([word for word in text.split() if word not in stop_words])  # Loại bỏ stopwords
    return text


# ===== Xử lý dữ liệu tổng hợp =====
def process_data(json_path, csv_path):
    df_list = []
    
    if json_path:
        df_json = load_json_data(json_path)
        df_list.append(df_json)
    
    if csv_path:
        df_csv = load_csv_data(csv_path)
        df_list.append(df_csv)
    
    if not df_list:
        raise ValueError("Both json_path and csv_path cannot be None")
    
    # Nếu chỉ có một dataset thì không cần concat
    df_combined = df_list[0] if len(df_list) == 1 else pd.concat(df_list, ignore_index=True)

     # Lọc bỏ các bản ghi có severity không hợp lệ (UNKNOWN)
    df_combined = df_combined[df_combined["severity"].isin(["LOW", "MEDIUM", "HIGH"])]

    # Tiền xử lý văn bản
    df_combined["description"] = df_combined["description"].apply(preprocess_text)

    # One-hot encoding cho accessVector
    df_combined = pd.get_dummies(df_combined, columns=["accessVector"], dtype=int)

    # Encode severity thành số (LOW=0, MEDIUM=1, HIGH=2)
    severity_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    df_combined["severity"] = df_combined["severity"].map(severity_mapping)

    return df_combined


# ===== Chạy xử lý dữ liệu =====
json_path = "W1/raw_cve_data.json"  
csv_path = "W3/dataset2/merged.csv"  

df_1 = load_json_data(json_path)  # Load JSON data into df_1
df_2 = load_csv_data(csv_path)  # Load CSV data into df_2
df_final = process_data(json_path, csv_path)

# Chia train/test
X = df_final.drop(columns=["cve_id", "severity"])  # Đặc trưng
y = df_final["severity"]  # Nhãn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra dữ liệu sau khi xử lý
print(df_final.head())

print(df_final.info())  # Xem thông tin cột và kiểu dữ liệu
print(df_final["severity"].value_counts())  # Xem phân bố nhãn

# print(df_final[df_final["severity"] == 3])  # Xem 34 mẫu bị lỗi (severity=3)

# print(df_1[df_1["cve_id"].isin(df_final[df_final["severity"] == 3]["cve_id"])])  # Xem thông tin của 34 mẫu bị lỗi
# print(df_2[df_2["cve_id"].isin(df_final[df_final["severity"] == 3]["cve_id"])]) # Xem thông tin của 34 mẫu bị lỗi


df_json = process_data(json_path, None)  # Chỉ JSON
df_csv = process_data(None, csv_path)  # Chỉ CSV

# Kiểm tra phân phối severity trong từng dataset
print("JSON dataset:")
print(df_json["severity"].value_counts(normalize=True))

print("\nCSV dataset:")
print(df_csv["severity"].value_counts(normalize=True))

# Kiểm tra các giá trị accessVector
print("\nAccessVector JSON:", df_json.columns)
print("AccessVector CSV:", df_csv.columns)


# Đồng nhất feature space: Cột accessVector_ADJACENT_NETWORK có trong CSV nhưng không có trong JSON.

# Chuẩn hóa tỷ lệ severity: Tránh thiên vị do phân phối khác nhau.

# 1️⃣ Chuẩn hóa dữ liệu (Xử lý NaN)
df_combined = process_data(json_path, csv_path)
df_combined.fillna("UNKNOWN", inplace=True)  

# 2️⃣ Tách tập dữ liệu theo nguồn
df_json_data = df_combined[df_combined["cve_id"].str.startswith("CVE-199")]  # Giả sử CVE từ JSON có pattern này
df_csv_data = df_combined[~df_combined["cve_id"].str.startswith("CVE-199")]

# 3️⃣ Chuyển đổi categorical features thành dạng số
existing_columns = ["access_vector", "access_complexity", "authentication"]
available_columns = [col for col in existing_columns if col in df_combined.columns]
df_encoded = pd.get_dummies(df_combined, columns=available_columns, dtype=int)


# 4️⃣ Train/Test Split cho từng dataset
X_json = df_encoded.loc[df_json_data.index].drop(columns=["cve_id", "severity"])
y_json = df_json_data["severity"]

X_csv = df_encoded.loc[df_csv_data.index].drop(columns=["cve_id", "severity"])
y_csv = df_csv_data["severity"]

X_train_json, X_test_json, y_train_json, y_test_json = train_test_split(X_json, y_json, test_size=0.2, random_state=42)
X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(X_csv, y_csv, test_size=0.2, random_state=42)

# 5️⃣ Train model riêng biệt
model_json = RandomForestClassifier(n_estimators=100, random_state=42)
model_json.fit(X_train_json, y_train_json)

model_csv = RandomForestClassifier(n_estimators=100, random_state=42)
model_csv.fit(X_train_csv, y_train_csv)

# 6️⃣ Kiểm tra từng model trên dataset tương ứng
print("📌 Model JSON trên JSON test set:")
print(classification_report(y_test_json, model_json.predict(X_test_json)))

print("📌 Model CSV trên CSV test set:")
print(classification_report(y_test_csv, model_csv.predict(X_test_csv)))

# 7️⃣ Cross-dataset Evaluation
print("📌 Model JSON trên CSV test set:")
print(classification_report(y_test_csv, model_json.predict(X_test_csv)))

print("📌 Model CSV trên JSON test set:")
print(classification_report(y_test_json, model_csv.predict(X_test_json)))
