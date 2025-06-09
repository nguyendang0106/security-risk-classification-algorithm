import pandas as pd
import numpy as np
import random 
import os
import glob
import matplotlib.pyplot as plt

random.seed(0)

drop_columns = [
    # Dataset Specific Information
    "Flow ID", 
    "Source IP", "Src IP", 
    "Source Port", "Src Port", 
    "Destination IP", "Dst IP",
    # Features Without Observed Variance
    "Bwd PSH Flags", 
    "Fwd URG Flags", 
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Byts/b Avg", 
    "Fwd Avg Packets/Bulk", "Fwd Pkts/b Avg", 
    "Fwd Avg Bulk Rate", "Fwd Blk Rate Avg",
    "Bwd Avg Bytes/Bulk", "Bwd Byts/b Avg", 
    "Bwd Avg Packets/Bulk", "Bwd Pkts/b Avg", 
    "Bwd Avg Bulk Rate", "Bwd Blk Rate Avg",
    # Duplicate Column
    "Fwd Header Length.1",

    # NaN
    "Protocol"

]

mapper = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Fwd Packets Length Total', 
    'Total Length of Fwd Packets': 'Fwd Packets Length Total',
    'TotLen Bwd Pkts': 'Bwd Packets Length Total',
    'Total Length of Bwd Packets': 'Bwd Packets Length Total', 
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min', 
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean', 
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max', 
    'Bwd Pkt Len Min': 'Bwd Packet Length Min', 
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std', 
    'Flow Byts/s': 'Flow Bytes/s', 
    'Flow Pkts/s': 'Flow Packets/s', 
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total', 
    'Fwd Header Len': 'Fwd Header Length', 
    'Bwd Header Len': 'Bwd Header Length', 
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s', 
    'Pkt Len Min': 'Packet Length Min', 
    'Min Packet Length': 'Packet Length Min',
    'Pkt Len Max': 'Packet Length Max', 
    'Max Packet Length': 'Packet Length Max',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std', 
    'Pkt Len Var': 'Packet Length Variance', 
    'FIN Flag Cnt': 'FIN Flag Count', 
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count', 
    'PSH Flag Cnt': 'PSH Flag Count', 
    'ACK Flag Cnt': 'ACK Flag Count', 
    'URG Flag Cnt': 'URG Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count', 
    'Pkt Size Avg': 'Avg Packet Size',
    'Average Packet Size': 'Avg Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size', 
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk', 
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate', 
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk', 
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate', 
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes', 
    'Subflow Bwd Pkts': 'Subflow Bwd Packets', 
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init Fwd Win Bytes', 
    'Init_Win_bytes_forward': 'Init Fwd Win Bytes',
    'Init Bwd Win Byts': 'Init Bwd Win Bytes', 
    'Init_Win_bytes_backward': 'Init Bwd Win Bytes',
    'Fwd Act Data Pkts': 'Fwd Act Data Packets',
    'act_data_pkt_fwd': 'Fwd Act Data Packets',
    'Fwd Seg Size Min': 'Fwd Seg Size Min',
    'min_seg_size_forward': 'Fwd Seg Size Min'
}

def plot_day(df):
    # Check if 'Timestamp' column exists before plotting
    if 'Timestamp' in df.columns:
        df.loc[df["Label"] == "Benign", 'Timestamp'].plot(style='.', color="lightgreen", label='Benign')
        for label in df.Label.unique():
            if label != 'Benign':
                df.loc[df["Label"] == label, 'Timestamp'].plot(style='.', label=label)
        plt.legend()
        plt.show()
    else:
        print("Warning: 'Timestamp' column not found for plotting.")

def clean_dataset(dataset, filetypes=['feather']):
    # Will search for all files in the dataset subdirectory 'original'
    for file in os.listdir(f'{dataset}/original'):
        print(f"------- {file} -------")
        try:  # Add try block for individual file processing
            df = pd.read_csv(f"{dataset}/original/{file}", skipinitialspace=True, encoding='latin')
            print(df["Label"].value_counts())
            print(f"Shape: {df.shape}")

            # Rename column names for uniform column names across files
            df.rename(columns=mapper, inplace=True)
            print("Columns after mapping:", df.columns.tolist())

            # Drop unrelevant columns
            df.drop(columns=drop_columns, inplace=True, errors="ignore")
            print("Columns after dropping:", df.columns.tolist())

            # Check if 'Timestamp' column exists before attempting to use it
            has_timestamp = 'Timestamp' in df.columns
            
            if has_timestamp:
                # Parse Timestamp column to pandas datetime
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                # Only process non-null timestamps
                df['Timestamp'] = df['Timestamp'].apply(
                    lambda x: x + pd.Timedelta(hours=12) if pd.notnull(x) and x.hour < 8 else x
                )
                df = df.sort_values(by=['Timestamp'])
            else:
                print(f"Warning: 'Timestamp' column not found in {file}. Skipping timestamp processing.")

            # Make Label column Categorical
            df['Label'].replace({'BENIGN': 'Benign'}, inplace=True)
            df['Label'] = df.Label.astype('category')

            # Parse Columns to correct dtype
            int_col = df.select_dtypes(include='integer').columns
            df[int_col] = df[int_col].apply(pd.to_numeric, errors='coerce', downcast='integer')
            float_col = df.select_dtypes(include='float').columns
            df[float_col] = df[float_col].apply(pd.to_numeric, errors='coerce', downcast='float')
            obj_col = df.select_dtypes(include='object').columns
            print(f'Columns with dtype == object: {obj_col}')
            for col in obj_col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with invalid data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            rows_before = df.shape[0]
            df.dropna(inplace=True)
            rows_after = df.shape[0]
            print(f"{rows_before - rows_after} invalid rows dropped")

            # Drop duplicate rows - exclude Label and Timestamp (if it exists) from the duplicate check
            subset_cols = df.columns.difference(['Label'])
            if has_timestamp:
                subset_cols = subset_cols.difference(['Timestamp'])
                
            rows_before_dedup = df.shape[0]
            df.drop_duplicates(inplace=True, subset=subset_cols)
            rows_after_dedup = df.shape[0]
            print(f"{rows_before_dedup - rows_after_dedup} duplicate rows dropped")
            
            print(df["Label"].value_counts())
            print(f"shape: {df.shape}\n")

            # Reset index
            df.reset_index(inplace=True, drop=True)

            # Plot resulting file only if timestamp exists
            if has_timestamp:
                plot_day(df)
            else:
                print("Skipping plot_day - no Timestamp column available")

            # Save to file
            if 'feather' in filetypes:
                df.to_feather(f'{dataset}/clean/{file}.feather')
            if 'parquet' in filetypes:
                df.to_parquet(f'{dataset}/clean/{file}.parquet', index=False)
                
        except Exception as e:
            print(f"ERROR processing file {file}: {e}")
            # Continue with next file instead of crashing
            
def aggregate_data(dataset, save=True, filetype='feather'):
    # Will search for all files in the 'clean' directory of the correct filetype and aggregate them
    all_data = pd.DataFrame()
    for file in glob.glob(f'{dataset}/clean/*.{filetype}'):
        print(file)
        df = pd.DataFrame()
        if filetype == 'feather':
            df = pd.read_feather(file)
        if filetype == 'parquet':
            df = pd.read_parquet(file)
        print(df.shape)
        print(f'{df["Label"].value_counts()}\n')
        # Replace deprecated append with concat
        all_data = pd.concat([all_data, df], ignore_index=True)
        
    print('ALL DATA')
    
    # Check if Timestamp exists in combined data
    has_timestamp = 'Timestamp' in all_data.columns
    
    # Define subset for duplicate check
    subset_cols = all_data.columns.difference(['Label'])
    if has_timestamp:
        subset_cols = subset_cols.difference(['Timestamp'])
        
    duplicates = all_data[all_data.duplicated(subset=subset_cols)]
    print('Removed duplicates after aggregating:')
    print(duplicates.Label.value_counts())
    print('Resulting Dataset')
    all_data.drop(duplicates.index, axis=0, inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    print(all_data.shape)
    print(f'{all_data["Label"].value_counts()}\n')
    
    if save:
        malicious = all_data[all_data.Label != 'Benign'].reset_index(drop=True)
        benign = all_data[all_data.Label == 'Benign'].reset_index(drop=True)
        if filetype == 'feather':
            all_data.to_feather(f'{dataset}/clean/all_data.feather')
            malicious.to_feather(f'{dataset}/clean/all_malicious.feather')
            benign.to_feather(f'{dataset}/clean/all_benign.feather')
        if filetype == 'parquet':
            all_data.to_parquet(f'{dataset}/clean/all_data.parquet', index=False)
            malicious.to_parquet(f'{dataset}/clean/all_malicious.parquet', index=False)
            benign.to_parquet(f'{dataset}/clean/all_benign.parquet', index=False)
            
if __name__ == "__main__":
    # Adjust for cleaning the correct dataset into the desired format
    
    # Needs directory with dataset name containing empty dir 'clean' and dir 'original' containing de csv's
    clean_dataset('W9/data2', filetypes=['feather', 'parquet'])
    aggregate_data('W9/data2', save=True, filetype='feather')
    aggregate_data('W9/data2', save=True, filetype='parquet')
