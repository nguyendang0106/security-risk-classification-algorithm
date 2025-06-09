import json
import pandas as pd

# Đọc dữ liệu từ file JSON đã lưu
with open("W1/raw_cve_data.json", "r") as f:
    data = json.load(f)

vulnerabilities = data.get("vulnerabilities", [])

cleaned_data = []

for item in vulnerabilities:
    cve_info = item.get("cve", {})
    cve_id = cve_info.get("id", "N/A")
    description = cve_info.get("descriptions", [{}])[0].get("value", "No description")
    severity = cve_info.get("metrics", {}).get("cvssMetricV2", [{}])[0].get("baseSeverity", "Unknown")
    cvss_score = cve_info.get("metrics", {}).get("cvssMetricV2", [{}])[0].get("cvssData", {}).get("baseScore", "N/A")
    references = [ref["url"] for ref in cve_info.get("references", [])]

    cleaned_data.append([cve_id, description, severity, cvss_score, ", ".join(references)])

df = pd.DataFrame(cleaned_data, columns=["CVE ID", "Description", "Severity", "CVSS Score", "References"])

df.to_csv("cleaned_cve_data.csv", index=False)

print("Dữ liệu đã được làm sạch và lưu vào 'cleaned_cve_data.csv'")