import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# Nạp biến môi trường từ file .env
load_dotenv()

# Khởi tạo và xác thực API
api = KaggleApi()
api.authenticate()

# Tên dataset giống như trên Kaggle
dataset_name = "solarmainframe/ids-intrusion-csv"

# Thư mục muốn lưu dataset
save_path = "W9/data2"

# Tải và giải nén dataset
api.dataset_download_files(dataset_name, path=save_path, unzip=True)

print(f"Dataset '{dataset_name}' đã được tải về thư mục: {save_path}")

# Install aws cli: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

# Fetch dataset files from S3 bucket:

# mkdir -p ~/Documents/onderzoek/experiment/real-time/data

# aws s3 cp --no-sign-request --region eu-west-3 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" ~/Documents/onderzoek/experiment/real-time/data --recursive

