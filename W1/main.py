import requests
import json


API_KEY = "d63c8c47-fdef-4de5-ba0d-bbffdd02c759"
URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

headers = {"apiKey": API_KEY}

response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()

    # Lưu dữ liệu thô vào file JSON để xử lý sau
    with open("raw_cve_data.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Dữ liệu đã được lưu vào 'raw_cve_data.json'")
else:
    print(f"Error: {response.status_code}")

