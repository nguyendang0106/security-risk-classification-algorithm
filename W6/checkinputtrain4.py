import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import gc

def format_data_string(data_array):
    """Formats a numpy array into the desired string format."""
    formatted_data = [f"{x:.8e}" for x in data_array]
    return '"data": [' + ','.join(formatted_data) + ']'

def main():
    print("Loading datasets...")
    try:
        # Ensure data directory exists
        if not os.path.exists('W6/data'):
             print("Error: Directory W6/data not found.")
             print("Please place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in W6/data/")
             return
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
        train_rows = len(train) # Store original number of training rows
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files exist in W6/data/")
        return

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
        # Note: train4.py calculates std/corr on the combined_data *after* label encoding
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat')['attack_cat'].nsmallest(7).index)
        exclude = list(set(lowCORR + lowSTD))
        if 'attack_cat' in exclude: exclude.remove('attack_cat')
        if 'severity_encoded' in exclude: exclude.remove('severity_encoded')
        print(f"Columns selected for PCA: {exclude}")

        # Apply PCA (fitted on combined data)
        pca = PCA(n_components=3)
        # Handle potential NaNs before PCA (using mean imputation for simplicity)
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
        # In train4.py, the scaling for the NN happens *after* this initial scaling,
        # but the RFE happens on data scaled this way. Let's stick to this scaling.
        # data_x = data_x.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)) # Manual scaling from train4.py
        # Using MinMaxScaler for consistency with the NN training part in train4.py
        scaler_all = MinMaxScaler()
        data_x_scaled = pd.DataFrame(scaler_all.fit_transform(data_x), columns=data_x.columns, index=data_x.index)
        print(f"Shape after scaling (data_x_scaled): {data_x_scaled.shape}")

        # --- Get the first sample from the original test portion ---
        test_start_index = train_rows
        if test_start_index >= len(data_x_scaled):
            print(f"\nError: Cannot get test sample, index ({test_start_index}) is out of bounds.")
            return

        # This is the sample after PCA, dur scaling, and MinMaxScaler on combined data
        first_test_sample_all_features_series = data_x_scaled.iloc[test_start_index]
        first_test_sample_all_features = first_test_sample_all_features_series.values

        # --- Generate Input for Model 1 (All Features) ---
        # This model in train4.py uses data_x_scaled split into train/val
        print("\n--- Sample Input for Model 1 (All Features) ---")
        output_string_all = format_data_string(first_test_sample_all_features)
        print(output_string_all)
        print(f"Number of features: {len(first_test_sample_all_features)}")
        print("Preprocessing: PCA, dur scaling, MinMaxScaler (fitted on combined data)")

        # --- Perform RFE to find selected features ---
        print("\n--- Performing RFE ---")
        # RFE in train4.py is fitted on X_train, y_train_attack which come from splitting data_x_scaled
        # Replicate the split used *before* RFE fitting in train4.py
        X_train_rfe_fit, _, y_train_attack_rfe_fit, _ = train_test_split(
            data_x_scaled, data_y_attack, test_size=0.20, random_state=42 # Matches the split before RFE in train4.py
        )
        print(f"Fitting RFE on data shape: {X_train_rfe_fit.shape}")
        rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10)
        rfe.fit(X_train_rfe_fit, y_train_attack_rfe_fit)
        selected_features_mask = rfe.support_
        selected_features = data_x_scaled.columns[selected_features_mask]
        print(f"Selected features by RFE ({len(selected_features)}): {list(selected_features)}")

        # --- Generate Input for Model 2 (RFE Features) ---
        # Model 2 uses data_x_selected, which is data_x filtered by RFE features, then scaled again.
        # For API input, we need the RFE features from the *already scaled* test sample.
        first_test_sample_rfe_features_series = first_test_sample_all_features_series[selected_features]
        first_test_sample_rfe_features = first_test_sample_rfe_features_series.values

        print("\n--- Sample Input for Model 2 (RFE Features) ---")
        output_string_rfe = format_data_string(first_test_sample_rfe_features)
        print(output_string_rfe)
        print(f"Number of features: {len(first_test_sample_rfe_features)}")
        print("Preprocessing: PCA, dur scaling, MinMaxScaler (fitted on combined data), RFE selection")

        # Clean up
        del train, test, combined_data, data_x, data_x_scaled, X_train_rfe_fit
        gc.collect()

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()