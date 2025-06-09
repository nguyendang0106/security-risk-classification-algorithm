import pandas as pd
import os
import glob

# Define file paths
benign_file_path = "W9/data2/test2/all_benign.parquet"
malicious_file_path = "W9/data2/test2/all_malicious.parquet"
output_dir = "W9/data2/test2/combined_mapped_data" # Directory to save the output
output_file_name = "combined_mapped_sampled_data.parquet"
output_parquet_path = os.path.join(output_dir, output_file_name)

samples_per_group = 10000 # Target samples per major group

# Define the mapping from original labels to broader groups
label_to_group_mapping = {
    'Benign': 'Benign',
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

def combine_map_and_sample_data(
    benign_path,
    malicious_path,
    output_path,
    mapping_dict,
    n_samples_per_group,
    default_group_name,
    random_state_seed=42
):
    """
    Reads benign and malicious Parquet files, combines them,
    maps labels to broader groups, samples data from each group,
    and saves the result to a new Parquet file.

    Args:
        benign_path (str): Path to the benign Parquet file.
        malicious_path (str): Path to the malicious Parquet file.
        output_path (str): Path to save the combined, mapped, and sampled Parquet file.
        mapping_dict (dict): Dictionary to map original labels to new groups.
        n_samples_per_group (int): The target number of samples for each group.
        default_group_name (str): Group name for labels not in the mapping_dict.
        random_state_seed (int): Random state for reproducibility of sampling.
    """
    dfs_to_combine = []

    # Read Benign data
    if os.path.exists(benign_path):
        print(f"Đang đọc dữ liệu Benign từ {benign_path}...")
        try:
            df_benign = pd.read_parquet(benign_path)
            # Ensure 'Label' column exists and set it to 'Benign' if necessary
            if 'Label' not in df_benign.columns and not df_benign.empty:
                 print(f"Cảnh báo: Tệp Benign {benign_path} không có cột 'Label'. Giả định tất cả là 'Benign'.")
                 df_benign['Label'] = 'Benign'
            elif 'Label' in df_benign.columns:
                # Verify all labels are indeed Benign or map them
                df_benign['Label'] = 'Benign' # Enforce Benign label
            print(f"  Đã đọc {len(df_benign)} dòng Benign.")
            dfs_to_combine.append(df_benign)
        except Exception as e:
            print(f"Lỗi khi đọc tệp Benign {benign_path}: {e}")
    else:
        print(f"Cảnh báo: Không tìm thấy tệp Benign tại {benign_path}.")

    # Read Malicious data
    if os.path.exists(malicious_path):
        print(f"Đang đọc dữ liệu Malicious từ {malicious_path}...")
        try:
            df_malicious = pd.read_parquet(malicious_path)
            if 'Label' not in df_malicious.columns and not df_malicious.empty:
                print(f"Lỗi: Tệp Malicious {malicious_path} không có cột 'Label'. Bỏ qua tệp này.")
            elif not df_malicious.empty:
                 print(f"  Đã đọc {len(df_malicious)} dòng Malicious.")
                 dfs_to_combine.append(df_malicious)
        except Exception as e:
            print(f"Lỗi khi đọc tệp Malicious {malicious_path}: {e}")
    else:
        print(f"Cảnh báo: Không tìm thấy tệp Malicious tại {malicious_path}.")

    if not dfs_to_combine:
        print("Không có dữ liệu để xử lý. Thoát.")
        return

    # Combine all dataframes
    df_combined = pd.concat(dfs_to_combine, ignore_index=True)
    print(f"\nTổng số dòng sau khi kết hợp: {len(df_combined)}")
    if 'Label' not in df_combined.columns:
        print("Lỗi: Cột 'Label' không tồn tại sau khi kết hợp dữ liệu.")
        return

    print("Phân phối nhãn gốc trong dữ liệu kết hợp:\n", df_combined['Label'].value_counts())

    # Create the new 'Group' column based on the mapping
    df_combined['Group'] = df_combined['Label'].apply(lambda x: mapping_dict.get(x, default_group_name))
    print("\nPhân phối sau khi ánh xạ sang các nhóm lớn:\n", df_combined['Group'].value_counts())

    # Group by the new 'Group' column and sample
    sampled_dfs_list = []
    for group_name_iter, group_df_iter in df_combined.groupby('Group'):
        if len(group_df_iter) >= n_samples_per_group:
            sampled_dfs_list.append(group_df_iter.sample(n=n_samples_per_group, random_state=random_state_seed, replace=False))
        else:
            print(f"Cảnh báo: Nhóm '{group_name_iter}' chỉ có {len(group_df_iter)} mẫu, ít hơn {n_samples_per_group} yêu cầu. Lấy tất cả mẫu có sẵn.")
            sampled_dfs_list.append(group_df_iter)

    if not sampled_dfs_list:
        print("Không có dữ liệu để lấy mẫu sau khi ánh xạ nhóm.")
        return

    final_sampled_df = pd.concat(sampled_dfs_list)

    # Shuffle the resulting dataframe
    final_sampled_df = final_sampled_df.sample(frac=1, random_state=random_state_seed).reset_index(drop=True)

    print(f"\nKích thước tập dữ liệu cuối cùng đã lấy mẫu và ánh xạ: {final_sampled_df.shape}")
    print("Phân phối nhóm cuối cùng:\n", final_sampled_df['Group'].value_counts())
    # print("Các nhãn gốc trong dữ liệu đã lấy mẫu:\n", final_sampled_df['Label'].value_counts().sort_index())

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Đã tạo (hoặc đã tồn tại) thư mục đầu ra: {os.path.dirname(output_path)}")

    try:
        final_sampled_df.to_parquet(output_path, index=False)
        print(f"\nĐã lưu thành công dữ liệu tổng hợp, ánh xạ và lấy mẫu vào {output_path}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào Parquet: {e}")

if __name__ == "__main__":
    # --- Tạo dữ liệu giả để kiểm thử nếu tệp không tồn tại ---
    # Bạn nên xóa hoặc bình luận phần này nếu bạn có tệp dữ liệu thực tế
    import random
    if not os.path.exists(benign_file_path):
        print(f"Tạo tệp giả cho {benign_file_path}")
        os.makedirs(os.path.dirname(benign_file_path), exist_ok=True)
        dummy_benign_data = {'Feature1': [random.random() for _ in range(2000)],
                             'Feature2': [random.randint(0,100) for _ in range(2000)],
                             'Label': ['Benign'] * 2000} # Explicitly add Label
        pd.DataFrame(dummy_benign_data).to_parquet(benign_file_path, index=False)

    if not os.path.exists(malicious_file_path):
        print(f"Tạo tệp giả cho {malicious_file_path}")
        os.makedirs(os.path.dirname(malicious_file_path), exist_ok=True)
        mal_labels_for_dummy = [
            'DDoS attacks-LOIC-HTTP', 'DDOS attack-HOIC', 'DoS attacks-Hulk',
            'Bot', 'Infilteration', 'SSH-Bruteforce', 'Brute Force -XSS', 'SQL Injection',
            'FTP-BruteForce', 'DoS attacks-SlowHTTPTest', 'Brute Force -Web'
        ]
        num_mal_rows = 5000
        dummy_mal_data = {'Feature1': [random.random() for _ in range(num_mal_rows)],
                          'Feature2': [random.randint(0,100) for _ in range(num_mal_rows)],
                          'Label': [random.choice(mal_labels_for_dummy) for _ in range(num_mal_rows)]}
        pd.DataFrame(dummy_mal_data).to_parquet(malicious_file_path, index=False)
    # --- Kết thúc phần tạo dữ liệu giả ---

    combine_map_and_sample_data(
        benign_path=benign_file_path,
        malicious_path=malicious_file_path,
        output_path=output_parquet_path,
        mapping_dict=label_to_group_mapping,
        n_samples_per_group=samples_per_group,
        default_group_name=default_group_for_unmapped
    )