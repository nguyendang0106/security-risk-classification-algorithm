import pandas as pd 

print("------------------------------------------------------------------------------------------------------------")

# For 2017 data
df_2017 = pd.read_parquet('W9/data1/clean/all_malicious.parquet')
print("2017 Protocol Info:")
print(df_2017['Protocol'].dtype)
print(f"NaNs: {df_2017['Protocol'].isnull().sum()}")
print(f"Unique values: {df_2017['Protocol'].unique()[:20]}") # Print some unique values

# For 2018 data
df_2018 = pd.read_parquet('W9/data2/clean/all_malicious.parquet')
print("\n2018 Protocol Info:")
print(df_2018['Protocol'].dtype)
print(f"NaNs: {df_2018['Protocol'].isnull().sum()}")
print(f"Unique values: {df_2018['Protocol'].unique()[:20]}")

print("------------------------------------------------------------------------------------------------------------")