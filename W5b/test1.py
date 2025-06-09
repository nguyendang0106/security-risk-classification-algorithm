import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

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
    "metrics_cvssMetricV2_0_cvssData_accessVector": "access_vector",
    "metrics_cvssMetricV2_0_cvssData_accessComplexity": "access_complexity",
    "metrics_cvssMetricV2_0_cvssData_authentication": "access_authentication",
    "metrics_cvssMetricV2_0_cvssData_confidentialityImpact": "impact_confidentiality",
    "metrics_cvssMetricV2_0_cvssData_integrityImpact": "impact_integrity",
    "metrics_cvssMetricV2_0_cvssData_availabilityImpact": "impact_availability",
    "weaknesses_0_description_0_value": "cwe_code",
    "descriptions_0_value": "summary",
})

df_csv = df_csv.rename(columns={"cwe_code": "cwe_code"})

# Ghép hai dataset
df = pd.concat([df_json, df_csv], ignore_index=True)

# Xử lý dữ liệu
df["published"] = pd.to_datetime(df["published"], errors="coerce")
df["lastModified"] = pd.to_datetime(df["lastModified"], errors="coerce")

df["days_since_published"] = (df["published"] - pd.Timestamp("1970-01-01")).dt.days
df["days_since_modified"] = (df["lastModified"] - pd.Timestamp("1970-01-01")).dt.days

# Điền giá trị thiếu
df.fillna({"cvss": df["cvss"].median()}, inplace=True)

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
label_encoders = {}
categorical_cols = ["access_vector", "access_complexity", "access_authentication", "impact_confidentiality", "impact_integrity", "impact_availability", "cwe_code"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Chuẩn hóa điểm CVSS
scaler = StandardScaler()
df["cvss_scaled"] = scaler.fit_transform(df[["cvss"]])

# Chọn các đặc trưng
features = ["days_since_published", "days_since_modified", "cvss_scaled"] + categorical_cols
X = df[features]

# Dự đoán điểm CVSS
y_reg = df["cvss"]
# Dự đoán mức độ nghiêm trọng
y_class = df["severity"]

# Chia tập train/test
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train mô hình hồi quy (CVSS score)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"Mean Absolute Error (CVSS score): {mae}")

# Train mô hình phân loại (Severity classification)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_class)
y_pred_class = clf.predict(X_test_cls)
print("Accuracy (Severity classification):", accuracy_score(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))
