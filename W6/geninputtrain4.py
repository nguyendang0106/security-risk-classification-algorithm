import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import gc
import random # Import random for sampling

def format_data_string(data_array):
    """Formats a numpy array into the desired string format."""
    formatted_data = [f"{x:.8e}" for x in data_array]
    return '"data": [' + ','.join(formatted_data) + ']'

def main(num_samples_to_generate=20): # Add parameter for number of samples
    print("Loading datasets...")
    output_dir = "W6/test_inputs"
    data_dir = "W6/data"

    # Define fixed output filenames
    output_file_all = os.path.join(output_dir, "model1_all_features_inputs.txt")
    output_file_rfe = os.path.join(output_dir, "model2_rfe_features_inputs.txt")

    try:
        # Ensure data directory exists
        if not os.path.exists(data_dir):
             print(f"Error: Data directory '{data_dir}' not found.")
             print(f"Please place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in {data_dir}/")
             return
        train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training-set.csv'))
        test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing-set.csv'))
        train_rows = len(train) # Store original number of training rows
        test_rows = len(test)   # Store original number of test rows
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Make sure the data files exist in {data_dir}/")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be appended to files in: {output_dir}")

    print("--- Replicating Preprocessing from train4.py ---")
    try:
        # Combine datasets
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)

        # Add severity_encoded column
        severity_map = {
            "Normal": 0, "Analysis": 2, "Backdoor": 3, "DoS": 3, "Exploits": 4,
            "Fuzzers": 1, "Generic": 2, "Reconnaissance": 1, "Shellcode": 4, "Worms": 4
        }
        combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)

        # Label encoding
        le_attack = LabelEncoder()
        combined_data['attack_cat'] = le_attack.fit_transform(combined_data['attack_cat'])
        combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
        combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
        combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

        # Feature analysis for PCA
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat')['attack_cat'].nsmallest(7).index)
        exclude = list(set(lowCORR + lowSTD))
        if 'attack_cat' in exclude: exclude.remove('attack_cat')
        if 'severity_encoded' in exclude: exclude.remove('severity_encoded')
        print(f"Columns selected for PCA: {exclude}")

        # Apply PCA (fitted on combined data)
        pca = PCA(n_components=3)
        pca_input_data = combined_data[exclude].fillna(combined_data[exclude].mean())
        dim_reduct = pca.fit_transform(pca_input_data)

        # Remove original features and add PCA results
        combined_data.drop(exclude, axis=1, inplace=True)
        dim_reduction = pd.DataFrame(dim_reduct, columns=['PCA1', 'PCA2', 'PCA3'], index=combined_data.index)
        combined_data = combined_data.join(dim_reduction)
        print(f"Shape after PCA: {combined_data.shape}")

        # Scale 'dur'
        if 'dur' in combined_data.columns:
            combined_data['dur'] = 10000 * combined_data['dur']

        # Prepare data for modeling
        data_x = combined_data.drop(['attack_cat', 'severity_encoded'], axis=1)
        data_y_attack = combined_data['attack_cat'] # Need this for RFE split

        print(f"\nShape of features (data_x) before scaling: {data_x.shape}")

        # Min-max scaling (Applied to the whole data_x in train4.py)
        print("\nApplying MinMaxScaler to all features (data_x)...")
        scaler = MinMaxScaler()
        data_x_scaled = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns, index=data_x.index)
        print(f"Shape after scaling (data_x_scaled): {data_x_scaled.shape}")

        # --- Perform RFE to find selected features (only needs to be done once) ---
        print("\n--- Performing RFE ---")
        X_train_rfe_fit, _, y_train_attack_rfe_fit, _ = train_test_split(
            data_x_scaled, data_y_attack, test_size=0.20, random_state=42
        )
        print(f"Fitting RFE on data shape: {X_train_rfe_fit.shape}")
        rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10)
        rfe.fit(X_train_rfe_fit, y_train_attack_rfe_fit)
        selected_features_mask = rfe.support_
        selected_features = data_x_scaled.columns[selected_features_mask]
        print(f"Selected features by RFE ({len(selected_features)}): {list(selected_features)}")

        # --- Select random samples from the original test portion ---
        test_indices_in_combined = range(train_rows, train_rows + test_rows)
        num_available_test = len(test_indices_in_combined)
        num_samples_to_select = min(num_samples_to_generate, num_available_test) # Don't try to select more than available

        if num_samples_to_select <= 0:
             print("\nError: No test samples available to select.")
             return

        print(f"\nSelecting {num_samples_to_select} random samples from the test set portion...")
        # Use random.sample to get unique indices without replacement
        selected_indices = random.sample(test_indices_in_combined, num_samples_to_select)

        # --- Generate and Append Input for Both Models for each selected sample ---
        print(f"\n--- Generating and Appending {num_samples_to_select} Input Samples ---")

        try:
            # Open files in append mode ('a')
            with open(output_file_all, 'a') as f_all, open(output_file_rfe, 'a') as f_rfe:
                for i, sample_index in enumerate(selected_indices):
                    print(f"Processing sample {i+1}/{num_samples_to_select} (Index: {sample_index})...")

                    # Get the full scaled sample
                    sample_all_features_series = data_x_scaled.iloc[sample_index]
                    sample_all_features = sample_all_features_series.values

                    # --- Model 1 (All Features) ---
                    output_string_all = format_data_string(sample_all_features)
                    f_all.write(output_string_all + '\n') # Append and add newline

                    # --- Model 2 (RFE Features) ---
                    sample_rfe_features_series = sample_all_features_series[selected_features]
                    sample_rfe_features = sample_rfe_features_series.values
                    output_string_rfe = format_data_string(sample_rfe_features)
                    f_rfe.write(output_string_rfe + '\n') # Append and add newline

            print(f"\nFinished appending {num_samples_to_select} samples to:")
            print(f"  - {output_file_all}")
            print(f"  - {output_file_rfe}")

        except IOError as e:
            print(f"\nError writing to output files: {e}")


        # Clean up
        del train, test, combined_data, data_x, data_x_scaled, X_train_rfe_fit
        gc.collect()

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() # Appends 20 samples by default
    