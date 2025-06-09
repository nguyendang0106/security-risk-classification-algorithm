import pandas as pd

# Đọc dữ liệu từ file CSV đã được làm sạch
df = pd.read_csv("W1/cleaned_cve_data.csv")

# Xử lý giá trị thiếu
# Thay thế NaN trong 'CVSS Score' bằng giá trị trung bình
df["CVSS Score"].fillna(df["CVSS Score"].mean(), inplace=True)
# Thay thế NaN trong 'References' bằng "No Reference"
df["References"].fillna("No Reference", inplace=True)

# Chuẩn hóa cột 'Severity' để đảm bảo dữ liệu thống nhất
df["Severity"] = df["Severity"].str.upper()

# Loại bỏ các khoảng trắng thừa trong 'References'
df["References"] = df["References"].apply(lambda x: ', '.join(set(x.split(', '))))

# Lưu dữ liệu đã tiền xử lý vào file mới
df.to_csv("W2/preprocessed_cve_data.csv", index=False)

print("Tiền xử lý dữ liệu hoàn tất! File đã lưu tại W2/preprocessed_cve_data.csv")
