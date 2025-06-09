import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

# Danh sách lưu dữ liệu
dataset = []

# Duyệt qua từng CVE
for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    metrics = cve.get("metrics", {})

    # Lấy mô tả tiếng Anh
    descriptions = cve.get("descriptions", [])
    description_text = ""
    for desc in descriptions:
        if desc["lang"] == "en":
            description_text = desc["value"]
            break  # Chỉ lấy mô tả tiếng Anh đầu tiên

    # Lấy chỉ số CVSS
    base_score = None
    exploit_score = None
    impact_score = None
    severity_label = None

    if "cvssMetricV2" in metrics:
        for metric in metrics["cvssMetricV2"]:
            severity_label = metric.get("baseSeverity", None)
            base_score = metric["cvssData"].get("baseScore", None)
            exploit_score = metric.get("exploitabilityScore", None)
            impact_score = metric.get("impactScore", None)
            break  # Lấy dữ liệu đầu tiên

    # Chuyển đổi nhãn severity thành số
    severity_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    severity_numeric = severity_map.get(severity_label, None)

    # Nếu thiếu dữ liệu, bỏ qua CVE này
    if description_text and severity_numeric is not None:
        dataset.append([description_text, base_score, exploit_score, impact_score, severity_numeric])

# Chuyển thành DataFrame
df = pd.DataFrame(dataset, columns=["description", "base_score", "exploit_score", "impact_score", "severity"])

# Chia tập train/test (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Lưu thành file CSV
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Số lượng mẫu train: {len(train_df)}")
print(f"Số lượng mẫu test: {len(test_df)}")
