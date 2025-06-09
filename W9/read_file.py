import pandas as pd 

print("------------------------------------------------------------------------------------------------------------")
print("Reading parquet...")
print("Dataset 1: CIC-IDS- 2017")
df = pd.read_parquet('W9/data1/clean/all_data.parquet')
# df = pd.read_csv('W3/dataset2/merged.csv')
# df = pd.read_parquet('W10/data2/test.parquet')
print(df.head())
print(df.shape)
print(df.info())
# In ra các nhãn có trong dữ liệu
print("Các loại nhãn trong dữ liệu:")
print(df['Label'].shape[0])  # số lượng nhãn
print(df['Label'].value_counts())  # tần suất mỗi nhãn
print("\nDanh sách nhãn duy nhất:")
print(df['Label'].unique().shape[0]) # số lượng nhãn duy nhất
print(df['Label'].unique())        # các nhãn duy nhất

# Lấy danh sách tên các cột (đặc trưng)
feature_names = df.columns.tolist()

# Tạo một DataFrame từ danh sách tên đặc trưng
features_df = pd.DataFrame(feature_names, columns=['Feature Name'])

# Lưu DataFrame này ra file Excel
excel_file_path = 'W9/feature_list_2017.xlsx'
features_df.to_excel(excel_file_path, index=False)

print(f"\nDanh sách các đặc trưng đã được lưu vào file: {excel_file_path}")


print("------------------------------------------------------------------------------------------------------------")


# print("Reading parquet...")
# print("Dataset 2: CSE-CIC-IDS-2018")
# df = pd.read_parquet('W9/data1/test2/combined_mapped_data/combined_mapped_sampled_data.parquet')
# # df = pd.read_parquet('W10/data2/infiltration_2018.parquet')
# print(df.head())
# print(df.shape)
# print(df.info())
# # In ra các nhãn có trong dữ liệu
# print("Các loại nhãn trong dữ liệu:")
# print(df['Label'].shape[0])  # số lượng nhãn
# print(df['Label'].value_counts())  # tần suất mỗi nhãn
# print("\nDanh sách nhãn duy nhất:")
# print(df['Label'].unique().shape[0]) # số lượng nhãn duy nhất
# print(df['Label'].unique())        # các nhãn duy nhất

print("------------------------------------------------------------------------------------------------------------")

# print("Reading parquet...")
# print("Dataset 1: CIC-IDS- 2017")
# df = pd.read_parquet('W9/dataComb/train/all_benign.parquet')
# # df = pd.read_parquet('W10/data2/test.parquet')
# print(df.head())
# print(df.shape)
# print(df.info())
# # In ra các nhãn có trong dữ liệu
# print("Các loại nhãn trong dữ liệu:")
# print(df['Label'].shape[0])  # số lượng nhãn
# print(df['Label'].value_counts())  # tần suất mỗi nhãn
# print("\nDanh sách nhãn duy nhất:")
# print(df['Label'].unique().shape[0]) # số lượng nhãn duy nhất
# print(df['Label'].unique())  

print("------------------------------------------------------------------------------------------------------------")

# print("Test selected features: CSE-CIC-IDS-2018")
# df = pd.read_parquet('W9/dataComb/train/all_malicious.parquet')
# # df = pd.read_parquet('W10/data2/test.parquet')
# print(df.head())
# print(df.shape)
# print(df.info())
# # In ra các nhãn có trong dữ liệu
# print("Các loại nhãn trong dữ liệu:")
# print(df['Label'].shape[0])  # số lượng nhãn
# print(df['Label'].value_counts())  # tần suất mỗi nhãn
# print("\nDanh sách nhãn duy nhất:")
# print(df['Label'].unique().shape[0]) # số lượng nhãn duy nhất
# print(df['Label'].unique())        # các nhãn duy nhất

# import pandas as pd 

# print("------------------------------------------------------------------------------------------------------------")

# # For 2017 data
# df_2017 = pd.read_parquet('W9/dataComb/train/all_benign.parquet')
# print("2017 Protocol Info:")
# print(df_2017['Protocol'].dtype)
# print(f"NaNs: {df_2017['Protocol'].isnull().sum()}")
# print(f"Unique values: {df_2017['Protocol'].unique()[:20]}") # Print some unique values

# # For 2018 data
# df_2018 = pd.read_parquet('W9/dataComb/train/all_malicious.parquet')
# print("\n2018 Protocol Info:")
# print(df_2018['Protocol'].dtype)
# print(f"NaNs: {df_2018['Protocol'].isnull().sum()}")
# print(f"Unique values: {df_2018['Protocol'].unique()[:20]}")

# print("------------------------------------------------------------------------------------------------------------")







































# print(df['y'].shape[0])  # số lượng nhãn
# print(df['y'].value_counts())  # tần suất mỗi nhãn
# print("\nDanh sách nhãn duy nhất:")
# print(df['y'].unique().shape[0]) # số lượng nhãn duy nhất
# print(df['y'].unique())        # các nhãn duy nhất


# print("\nReading feather...")
# df = pd.read_feather('W9/data1/clean/all_malicious.feather')
# print(df.head())
# print(df.shape)
# print(df.info())

# # Đọc lại 2 file
# df_parquet = pd.read_parquet('W9/data1/clean/all_malicious.parquet')
# df_feather = pd.read_feather('W9/data1/clean/all_malicious.feather')

# # Tìm các dòng có trong parquet nhưng không có trong feather
# diff_rows = pd.concat([df_parquet, df_feather, df_feather]).drop_duplicates(keep=False)
# print(diff_rows)
# print("Số dòng khác biệt:", diff_rows.shape)



# --- Performing Feature Selection ---
# Training temporary RF for feature selection...
# Temporary RF trained.
# Selected top 15 feature indices: [14  8 37 48  6 13  5 56 50 34 12 15 58 39 54]
# Selected top 15 feature names: ['Flow Bytes/s', 'Fwd Packet Length Mean', 
#                                 'Packet Length Mean', 'Avg Packet Size', 
#                                 'Fwd Packet Length Max', 'Bwd Packet Length Std', 
#                                 'Bwd Packets Length Total', 'Init Bwd Win Bytes', 
#                                 'Avg Bwd Segment Size', 'Bwd Packets/s', 
#                                 'Bwd Packet Length Mean', 'Flow Packets/s', 
#                                 'Fwd Seg Size Min', 'Packet Length Variance', 
#                                 'Subflow Bwd Bytes']
