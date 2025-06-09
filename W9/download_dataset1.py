import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Nạp biến môi trường từ file .env
load_dotenv()

# Khởi tạo và xác thực API
api = KaggleApi()
api.authenticate()

# Tên dataset giống như trên Kaggle
dataset_name = "chethuhn/network-intrusion-dataset"

# Thư mục muốn lưu dataset
save_path = "W9/data1"

# Tải và giải nén dataset
api.dataset_download_files(dataset_name, path=save_path, unzip=True)

print(f"Dataset '{dataset_name}' đã được tải về thư mục: {save_path}")


# http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/