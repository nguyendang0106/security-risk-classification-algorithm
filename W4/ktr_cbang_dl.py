import pandas as pd
import json

# Đọc file JSON
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

# Trích xuất nhãn baseSeverity
labels = []
for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    metrics = cve.get("metrics", {})
    
    # Kiểm tra nếu có cvssMetricV2
    if "cvssMetricV2" in metrics:
        for metric in metrics["cvssMetricV2"]:
            if "baseSeverity" in metric:
                labels.append(metric["baseSeverity"])

# Đưa vào DataFrame
df = pd.DataFrame({"baseSeverity": labels})

# Đếm số lượng mẫu trong từng lớp
class_counts = df["baseSeverity"].value_counts()
print("Số lượng mẫu trong từng lớp:")
print(class_counts)


# Số lượng mẫu trong từng lớp:
# baseSeverity
# HIGH      999
# MEDIUM    747
# LOW       220
# Name: count, dtype: int64


#  Nhận xét
# Lớp LOW có 220 mẫu, thấp hơn đáng kể so với MEDIUM (747 mẫu) và HIGH (999 mẫu).
# Điều này có thể làm cho mô hình học kém về lớp LOW, dẫn đến recall thấp như kết quả trước đó.
