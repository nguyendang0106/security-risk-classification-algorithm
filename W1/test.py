import json  

# Đọc file JSON nếu đã lưu  
with open("W1/raw_cve_data.json", "r") as f:  
    data = json.load(f)  

print(type(data))  # Kiểm tra kiểu dữ liệu: dict

print(data.keys())  # In ra tất cả các khóa trong dữ liệu

print(json.dumps(data["totalResults"], indent=2)) # Số lượng kết quả

print(f"Số lượng CVE: {len(data['vulnerabilities'])}") # Số lượng : 2000

print(json.dumps(data["vulnerabilities"][0], indent=2))  # In ra thông tin của một CVE đầu tiên

print(data["vulnerabilities"][0]["cve"].keys())  # In ra tất cả các khóa trong "cve"


# Tạo một tập hợp để lưu trữ các giá trị khác nhau của baseSeverity
severity_levels = set()

# Duyệt qua tất cả CVE trong danh sách "vulnerabilities"
for entry in data["vulnerabilities"]:
    cve = entry.get("cve", {})
    metrics = cve.get("metrics", {})
    
    # Kiểm tra nếu có cvssMetricV2
    if "cvssMetricV2" in metrics:
        for metric in metrics["cvssMetricV2"]:
            if "baseSeverity" in metric:
                severity_levels.add(metric["baseSeverity"])

# In ra danh sách các nhãn baseSeverity tìm thấy
print("Các nhãn baseSeverity có trong dữ liệu:", severity_levels)


# {
#   "cve": {
#     "id": "CVE-1999-0095",
#     "sourceIdentifier": "cve@mitre.org",
#     "published": "1988-10-01T04:00:00.000",
#     "lastModified": "2024-11-20T23:27:50.607",
#     "vulnStatus": "Modified",
#     "descriptions": [...],
#     "metrics": {...},
#     "weaknesses": [...],
#     "configurations": [...],
#     "references": [...]
#   }
# }

# "id" là mã CVE duy nhất.
# "published" và "lastModified" là ngày tháng.
# "descriptions" chứa mô tả bằng nhiều ngôn ngữ.
# "metrics" chứa điểm CVSS (chỉ số đánh giá mức độ nghiêm trọng của lỗ hổng).
# "weaknesses" liên quan đến các điểm yếu hệ thống.
# "configurations" là danh sách các hệ thống bị ảnh hưởng.
# "references" là danh sách tài liệu liên quan.


# ID CVE (cve["id"]) → (Có thể làm index, không dùng để train)
# Mô tả lỗ hổng (cve["descriptions"]) → (Có thể dùng NLP để trích xuất thông tin)
# Chỉ số CVSS (cve["metrics"]) → (Quan trọng cho đánh giá rủi ro)
# Mức độ nghiêm trọng (baseSeverity) → (Mục tiêu dự đoán hoặc làm feature)
# Weaknesses (CWE) (cve["weaknesses"]) → (Nhóm lỗ hổng liên quan)
# Cấu hình bị ảnh hưởng (configurations) (cpeMatch) → (Có thể chuẩn hóa thành binary feature)