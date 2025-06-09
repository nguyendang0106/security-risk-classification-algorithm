import pandas as pd
import os
import random # for dummy data generation

# Define file paths
input_parquet_path = "W9/data2/clean/all_data.parquet"
# Naming the output file to reflect the specific mapping and sampling
output_parquet_path = "W9/dataTest2018/sampled_CSE-CIC-IDS-2018_mapped_1000_per_group.parquet"
samples_per_group = 1000

# Define the mapping from original labels to broader groups as per your request
# Based on the provided data analysis and mapping
label_to_group_mapping = {
    'Benign': 'Benign',  # Assuming 'Benign' is the exact label string from your data
    'DDoS attacks-LOIC-HTTP': '(D)DOS',
    'DDOS attack-HOIC': '(D)DOS',
    'DoS attacks-Hulk': '(D)DOS',
    'DoS attacks-GoldenEye': '(D)DOS',
    'DoS attacks-Slowloris': '(D)DOS',
    'DDOS attack-LOIC-UDP': '(D)DOS',
    'DoS attacks-SlowHTTPTest': '(D)DOS',
    'Bot': 'Botnet',
    'SSH-Bruteforce': 'Brute Force',
    'FTP-BruteForce': 'Brute Force',
    'Brute Force -Web': 'Brute Force',
    'Brute Force -XSS': 'Web Attack',
    'SQL Injection': 'Web Attack',
    'Infilteration': 'Unknown'
}
# Default group for any labels not explicitly in the mapping above
default_group_for_unmapped = 'Unknown'

def create_sampled_grouped_dataset(input_path, output_path, mapping_dict, n_samples_per_group, default_group_name, random_state_seed=42):
    """
    Reads a Parquet file, maps labels to broader groups, samples data from each group,
    and saves the result to a new Parquet file.

    Args:
        input_path (str): Path to the input Parquet file.
        output_path (str): Path to save the sampled and mapped Parquet file.
        mapping_dict (dict): Dictionary to map original labels to new groups.
        n_samples_per_group (int): The target number of samples for each group.
        default_group_name (str): Group name for labels not in the mapping_dict.
        random_state_seed (int): Random state for reproducibility of sampling.
    """
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy tệp đầu vào tại {input_path}")
        # Create dummy data if file not found, for testing purposes
        print(f"Cảnh báo: {input_path} không tìm thấy. Tạo tệp giả để minh họa.")
        num_rows = 2000000  # More rows for dummy data
        original_labels_for_dummy = [
            'Benign', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk',
            'Bot', 'Infilteration', 'SSH-Bruteforce', 'DoS attacks-GoldenEye',
            'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP', 'Brute Force -Web',
            'Brute Force -XSS', 'SQL Injection', 'FTP-BruteForce', 'DoS attacks-SlowHTTPTest'
        ]
        # Create a somewhat representative distribution for dummy data
        dummy_labels_list = []
        # High count for Benign
        dummy_labels_list.extend(['Benign'] * 1000000)
        # Moderate for DDoS types
        for lbl in ['DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk']:
            dummy_labels_list.extend([lbl] * 50000)
        dummy_labels_list.extend(['Bot'] * 50000)
        dummy_labels_list.extend(['Infilteration'] * 40000)
        dummy_labels_list.extend(['SSH-Bruteforce'] * 30000)
        # Smaller for others
        for lbl in ['DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP',
                    'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection', 'FTP-BruteForce',
                    'DoS attacks-SlowHTTPTest']:
            dummy_labels_list.extend([lbl] * 500)

        # Fill remaining if any, or truncate
        if len(dummy_labels_list) < num_rows:
            dummy_labels_list.extend(random.choices(original_labels_for_dummy, k=num_rows - len(dummy_labels_list)))
        else:
            dummy_labels_list = dummy_labels_list[:num_rows]
        random.shuffle(dummy_labels_list)

        dummy_data = {
            'Label': dummy_labels_list,
            # Add a few dummy feature columns
            'Feature1': [random.random() for _ in range(num_rows)],
            'Feature2': [random.randint(0, 100) for _ in range(num_rows)],
            'Timestamp': pd.to_datetime(['1970-01-01']*num_rows) # Dummy timestamp
        }
        # Add other columns present in the original data structure if known, e.g., Destination Port
        if 'Destination Port' not in dummy_data: # Example, add more if needed
             dummy_data['Destination Port'] = [random.randint(0,65535) for _ in range(num_rows)]


        dummy_df_to_save = pd.DataFrame(dummy_data)
        dummy_dir = os.path.dirname(input_path)
        if dummy_dir and not os.path.exists(dummy_dir):
            os.makedirs(dummy_dir)
        dummy_df_to_save.to_parquet(input_path, index=False)
        print(f"Tệp giả đã được tạo tại {input_path}")
        # return # Optionally stop if dummy data was created and real data is expected

    print(f"Đang đọc dữ liệu từ {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Lỗi khi đọc tệp Parquet: {e}")
        return

    print("\nKích thước tập dữ liệu gốc:", df.shape)
    if 'Label' not in df.columns:
        print("Lỗi: Cột 'Label' không tồn tại trong tập dữ liệu.")
        return
    print("Phân phối nhãn gốc:\n", df['Label'].value_counts())

    # Tạo cột 'Group' mới dựa trên ánh xạ
    df['Group'] = df['Label'].apply(lambda x: mapping_dict.get(x, default_group_name))

    print("\nPhân phối sau khi ánh xạ sang các nhóm:\n", df['Group'].value_counts())

    # Nhóm theo cột 'Group' mới và lấy mẫu
    sampled_dfs_list = []
    for group_name_iter, group_df_iter in df.groupby('Group'):
        if len(group_df_iter) >= n_samples_per_group:
            # If group has enough samples, sample n_samples_per_group
            sampled_dfs_list.append(group_df_iter.sample(n=n_samples_per_group, random_state=random_state_seed, replace=False))
        else:
            # If group has fewer samples than n_samples_per_group, take all available samples
            print(f"Cảnh báo: Nhóm '{group_name_iter}' chỉ có {len(group_df_iter)} mẫu, ít hơn {n_samples_per_group} yêu cầu. Lấy tất cả mẫu có sẵn.")
            sampled_dfs_list.append(group_df_iter)

    if not sampled_dfs_list:
        print("Không có dữ liệu để lấy mẫu hoặc không có cột 'Group' được tạo.")
        return

    final_sampled_df = pd.concat(sampled_dfs_list)

    # Xáo trộn DataFrame kết quả (tùy chọn, nhưng nên làm)
    final_sampled_df = final_sampled_df.sample(frac=1, random_state=random_state_seed).reset_index(drop=True)

    print(f"\nKích thước tập dữ liệu đã lấy mẫu và ánh xạ: {final_sampled_df.shape}")
    print("Phân phối nhóm đã lấy mẫu và ánh xạ:\n", final_sampled_df['Group'].value_counts())
    # print("Các nhãn gốc trong dữ liệu đã lấy mẫu:\n", final_sampled_df['Label'].value_counts().sort_index())


    # Đảm bảo thư mục đầu ra tồn tại
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Đã tạo thư mục: {output_directory}")

    try:
        final_sampled_df.to_parquet(output_path, index=False)
        print(f"\nĐã lưu thành công dữ liệu đã lấy mẫu và ánh xạ vào {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào Parquet: {e}")

if __name__ == "__main__":
    create_sampled_grouped_dataset(
        input_path=input_parquet_path,
        output_path=output_parquet_path,
        mapping_dict=label_to_group_mapping,
        n_samples_per_group=samples_per_group,
        default_group_name=default_group_for_unmapped
    )