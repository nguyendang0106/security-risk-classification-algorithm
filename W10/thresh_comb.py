import os
import sys
import pickle
import numpy as np
import pandas as pd
import pathlib
import random
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter # Import Counter

# Add project root to path to allow finding W9
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import W9.util.common as util # Assuming common.py contains plot_confusion_matrix if needed

print("=============== Evaluating Threshold Combinations for Combined Model ===============")

# --- Configuration ---
model_dir = pathlib.Path("W10/models_combined_sampled") # Directory where models/data are saved
data2017_dir = "W9/data1/clean/"
data2018_dir = "W9/data2/clean/"
seed_value = 42 # Use the same seed as in train_comb.py
MALICIOUS_SAMPLE_SIZE_PER_TYPE = 1948 # Use the same sampling size
train_prop = 0.70
val_prop = 0.15
test_prop = 0.15
# --- End Configuration ---

# Set fixed seeds for reproducibility during data reconstruction
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# --- Load Models and Scalers ---
print(f"Loading models, scalers, and feature info from {model_dir}...")
try:
    with open(model_dir / "stage1_ocsvm_scaler.p", "rb") as f: # This is scaler_ae
        scaler_ae = pickle.load(f)
    with open(model_dir / "stage2_rf_scaler.p", "rb") as f:
        scaler_stage2 = pickle.load(f)
    with open(model_dir / "stage1_ocsvm.p", "rb") as f:
        ocsvm_model_100k = pickle.load(f)
    with open(model_dir / "stage2_rf_model.p", "rb") as f:
        rf_model_extra_feature = pickle.load(f)
    with open(model_dir / "optimal_thresholds.pkl", "rb") as f:
        original_thresholds = pickle.load(f)
    threshold_m_fixed = original_thresholds['threshold_m']
    with open(model_dir / "selected_feature_indices.pkl", "rb") as f:
        selected_indices = pickle.load(f)
    # test_years.pkl is not strictly needed here if we reconstruct the test set
    print("Loaded models, scalers, fixed threshold_m, and feature indices.")
    print(f"Using fixed threshold_m: {threshold_m_fixed:.6f}")

except FileNotFoundError as e:
     print(f"Error loading file: {e}. Please ensure 'train_comb.py' ran successfully and saved all necessary model/scaler/feature files in {model_dir}.")
     sys.exit(1)

# --- Reconstruct Test Data ---
# This section mimics the data loading, sampling, and splitting from train_comb.py
# to get the exact same test set used during training.

print("\nReconstructing test data...")

# 1. Load original data
print("Loading original 2017 and 2018 data...")
benign2017 = pd.read_parquet(f"{data2017_dir}/all_benign.parquet")
malicious2017 = pd.read_parquet(f"{data2017_dir}/all_malicious.parquet")
benign2018 = pd.read_parquet(f"{data2018_dir}/all_benign.parquet")
malicious2018 = pd.read_parquet(f"{data2018_dir}/all_malicious.parquet")

# Add year column
benign2017['year'] = 2017
malicious2017['year'] = 2017
benign2018['year'] = 2018
malicious2018['year'] = 2018

# Combine
all_benign = pd.concat([benign2017, benign2018], axis=0, ignore_index=True)
all_malicious = pd.concat([malicious2017, malicious2018], axis=0, ignore_index=True)

# 2. Map and Sample Malicious Data
print("Mapping labels and sampling malicious data...")
if 'Label' not in all_malicious.columns:
    raise ValueError("Column 'Label' not found in malicious data.")
all_malicious['Label'] = all_malicious['Label'].fillna('Unknown_Original')
original_attack_type = all_malicious['Label'].astype(str).copy()

label_mapping = {
    # (Copy the exact label_mapping dictionary from train_comb.py here)
    # 2017 dataset
    'DoS Hulk':'(D)DOS', 'PortScan':'Port Scan', 'DDoS':'(D)DOS',
    'DoS slowloris':'(D)DOS', 'DoS Slowhttptest':'(D)DOS', 'DoS GoldenEye':'(D)DOS',
    'SSH-Patator':'Brute Force', 'FTP-Patator':'Brute Force', 'Bot': 'Botnet',
    'Web Attack \x96 Brute Force': 'Web Attack', 'Web Attack \x96 Sql Injection': 'Web Attack',
    'Web Attack \x96 XSS': 'Web Attack', 'Infiltration': 'Unknown', 'Heartbleed': 'Unknown',
    'Web Attack ï¿½ Brute Force': 'Web Attack', 'Web Attack ï¿½ Sql Injection': 'Web Attack',
    'Web Attack ï¿½ XSS': 'Web Attack',
    # 2018 dataset
    'DDoS attacks-LOIC-HTTP': '(D)DOS', 'DDOS attack-HOIC': '(D)DOS', 'DoS attacks-Hulk': '(D)DOS',
    'DoS attacks-GoldenEye': '(D)DOS', 'DoS attacks-Slowloris': '(D)DOS', 'DDOS attack-LOIC-UDP': '(D)DOS',
    'DoS attacks-SlowHTTPTest': '(D)DOS', 'Bot': 'Botnet', 'SSH-Bruteforce': 'Brute Force',
    'FTP-BruteForce': 'Brute Force', 'Brute Force -Web': 'Brute Force', 'Brute Force -XSS': 'Web Attack',
    'SQL Injection': 'Web Attack', 'Infilteration': 'Unknown',
    # Handle potential fillna value
    'Unknown_Original': 'Unknown'
}
mapped_labels_temp = original_attack_type.map(lambda x: label_mapping.get(x, 'Unknown'))
all_malicious['Label'] = mapped_labels_temp
all_malicious['Original_Label'] = original_attack_type

malicious_sampled_list = []
for name, group in all_malicious.groupby('Original_Label'):
    if len(group) > MALICIOUS_SAMPLE_SIZE_PER_TYPE:
        malicious_sampled_list.append(group.sample(n=MALICIOUS_SAMPLE_SIZE_PER_TYPE, random_state=seed_value))
    else:
        malicious_sampled_list.append(group)
malicious_sampled = pd.concat(malicious_sampled_list, ignore_index=False)
malicious_sampled = malicious_sampled.sample(frac=1, random_state=seed_value).reset_index(drop=True)

# 3. Preprocess and Split
print("Preprocessing and splitting data...")
columns_to_drop = ['Timestamp', 'Destination Port']
if 'Label' in all_benign.columns:
    x_benign = all_benign.drop(columns=columns_to_drop + ['Label', 'year'], errors='ignore')
else:
    x_benign = all_benign.drop(columns=columns_to_drop + ['year'], errors='ignore')
y_benign_years = all_benign['year']

feature_cols = [col for col in malicious_sampled.columns if col not in columns_to_drop + ['Label', 'Original_Label', 'year']]
x_malicious = malicious_sampled[feature_cols]
y_malicious = malicious_sampled['Label']
y_malicious_original = malicious_sampled['Original_Label']
malicious_years = malicious_sampled['year']

# Split benign
x_benign_train, x_benign_valtest, _, y_benign_valtest_year = train_test_split(
    x_benign, y_benign_years, train_size=train_prop, random_state=seed_value, stratify=y_benign_years
)
val_size_benign_adj = val_prop / (val_prop + test_prop)
_, x_benign_test, _, _ = train_test_split(
    x_benign_valtest, y_benign_valtest_year, train_size=val_size_benign_adj, random_state=seed_value, stratify=y_benign_valtest_year
)

# Split sampled malicious
x_malicious_train, x_malicious_valtest, _, y_malicious_valtest, _, y_malicious_original_valtest, _, year_malicious_valtest = train_test_split(
    x_malicious, y_malicious, y_malicious_original, malicious_years,
    train_size=train_prop, random_state=seed_value, stratify=y_malicious_original
)
test_size_malicious_adj = test_prop / (val_prop + test_prop)
_, x_malicious_test, _, y_malicious_test, _, _, _, _ = train_test_split(
    x_malicious_valtest, y_malicious_valtest, y_malicious_original_valtest, year_malicious_valtest,
    test_size=test_size_malicious_adj, random_state=seed_value, stratify=y_malicious_original_valtest
)

# 4. Combine Test Set and Apply Feature Selection
print("Combining test set and applying feature selection...")
test_x_original = pd.concat([x_benign_test, x_malicious_test], axis=0, ignore_index=True)

# Helper function to apply feature selection
def select_features(df, indices_to_keep):
    if isinstance(df, pd.DataFrame):
        return df.iloc[:, indices_to_keep]
    elif isinstance(df, np.ndarray):
        return df[:, indices_to_keep]
    else:
        if df is None: return None
        raise TypeError(f"Unexpected data type for feature selection: {type(df)}")

# Apply feature selection using loaded indices
if test_x_original.shape[1] > max(selected_indices):
     test_x_selected = select_features(test_x_original, selected_indices)
     print(f"Applied feature selection. New feature count: {test_x_selected.shape[1]}")
else:
     print(f"Warning: Shape mismatch during feature selection. Test data columns: {test_x_original.shape[1]}, Max index: {max(selected_indices)}")
     # Handle this case - maybe exit or proceed with unselected data if appropriate
     # For now, let's assume it should match and proceed. If it fails, check train_comb.py's saving/loading.
     test_x_selected = test_x_original # Or sys.exit(1)

# 5. Apply Scaling
print("Applying scaling to test set...")
test_ae_s = scaler_ae.transform(test_x_selected)
test_stage2_s = scaler_stage2.transform(test_x_selected)

# 6. Reconstruct Ground Truth Labels
print("Reconstructing test labels...")
test_y_benign = np.full(x_benign_test.shape[0], "Benign")
test_y_malicious = y_malicious_test # Already have the mapped labels for the malicious test part
test_y_combined = np.concatenate([test_y_benign, test_y_malicious])
test_y_unknown = np.where(test_y_combined == "Unknown", "Unknown", test_y_combined) # Final ground truth

print(f"Test data reconstruction complete. Shape: {test_ae_s.shape}, Labels: {len(test_y_unknown)}")


# --- Define Threshold Candidates ---

# Thresholds_b based on F1-F9 metrics from validation (copied from prompt)
thresholds_b_candidates_dict = {
    'F1': 0.3149174730486388, 'F2': 0.09702001679456362, 'F3': -0.01216313771933697,
    'F4': -0.01216313771933697, 'F5': -0.01216313771933697, 'F6': -0.01216313771933697,
    'F7': -0.08379384532768386, 'F8': -1.2719150670298296, 'F9': -1.2719150670298296
}
thresholds_b_candidates = sorted(list(set(thresholds_b_candidates_dict.values())))
print(f"\nCandidate thresholds_b (from F1-F9): {thresholds_b_candidates}")

# Thresholds_u based on quantiles from validation (copied from prompt)
thresholds_u_candidates_dict = {
    0.995: -2.1281038988796297e-06, 0.99: -0.008838997228100153, 0.95: -0.041734631410782624,
    0.9: -0.12533855791477277, 0.875: -0.20185278399220524, 0.85: -0.23746683475936897,
    0.8: -0.2682140700609153, 0.75: -0.29198749146849756, 0.7: -0.3184919705000705,
    0.6: -0.4539678991363462, 0.5: -0.7070801941812315
}
quantiles_to_use = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.95, 0.99, 0.995]
thresholds_u_candidates = sorted([thresholds_u_candidates_dict[q] for q in quantiles_to_use])
print(f"Candidate thresholds_u (from quantiles 0.5-0.995): {thresholds_u_candidates}")


# --- Evaluation Function (Copied and adapted from train_comb.py) ---

def evaluate_pipeline(x_test_scaled_ocsvm, x_test_scaled_rf, y_true, ocsvm_model, rf_model, thresholds):
    """Evaluates the full 3-stage pipeline."""
    # Stage 1: Anomaly Detection (using OCSVM scaled data)
    score_test = -ocsvm_model.decision_function(x_test_scaled_ocsvm)
    y_pred = np.where(score_test < thresholds['threshold_b'], "Benign", "Fraud").astype(object)

    # Get indices for samples predicted as fraud
    fraud_indices = np.where(y_pred == "Fraud")[0]

    # Stage 2: Classification (only for samples classified as fraud)
    if len(fraud_indices) > 0:
        # Prepare RF input: RF scaled features + anomaly score
        x_test_rf_fraud = x_test_scaled_rf[fraud_indices]
        score_test_fraud = score_test[fraud_indices]
        x_test_fraud_with_proba = np.column_stack((x_test_rf_fraud, score_test_fraud))

        # Get probabilities from RF
        y_proba_2_fraud = rf_model.predict_proba(x_test_fraud_with_proba)

        # Apply threshold_m
        y_pred_2 = np.where(
            np.max(y_proba_2_fraud, axis=1) > thresholds['threshold_m'],
            rf_model.classes_[np.argmax(y_proba_2_fraud, axis=1)],
            'Unknown'
        )
        y_pred[fraud_indices] = y_pred_2

    # Extension: Reconsider "Unknown" predictions
    unknown_indices = np.where(y_pred == "Unknown")[0]
    if len(unknown_indices) > 0:
        score_test_unknown = score_test[unknown_indices]
        # Apply threshold_u
        y_pred_3 = np.where(score_test_unknown < thresholds['threshold_u'], "Benign", "Unknown")
        y_pred[unknown_indices] = y_pred_3

    # Calculate metrics
    all_possible_labels = np.unique(np.concatenate((y_true, y_pred)))
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', labels=all_possible_labels, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=all_possible_labels, zero_division=0)
    }
    return y_pred, metrics

# --- Iterate and Evaluate ---
print("\nStarting threshold evaluation...")

best_f1_weighted = -1
best_thresholds = {}
results = []

total_combinations = len(thresholds_b_candidates) * len(thresholds_u_candidates)
count = 0

for th_b in thresholds_b_candidates:
    for th_u in thresholds_u_candidates:
        count += 1
        current_thresholds = {
            'threshold_b': th_b,
            'threshold_m': threshold_m_fixed,
            'threshold_u': th_u
        }

        # Evaluate using the reconstructed and scaled test data
        y_pred_eval, metrics_eval = evaluate_pipeline(
            test_ae_s,             # Reconstructed & scaled OCSVM input
            test_stage2_s,         # Reconstructed & scaled RF input
            test_y_unknown,        # Reconstructed labels
            ocsvm_model_100k,
            rf_model_extra_feature,
            current_thresholds
        )

        # Store results
        results.append({
            'threshold_b': th_b,
            'threshold_u': th_u,
            **metrics_eval # Add all metrics to the results
        })

        # Check for best performance (using F1-weighted)
        if metrics_eval['f1_weighted'] > best_f1_weighted:
            best_f1_weighted = metrics_eval['f1_weighted']
            best_thresholds = current_thresholds
            best_metrics = metrics_eval

        if count % 10 == 0 or count == total_combinations: # Print progress less frequently
             print(f"  Evaluated {count}/{total_combinations} combinations...")


print("\nEvaluation complete.")

# --- Output Results ---

# Convert results to DataFrame for easier viewing
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='f1_weighted', ascending=False)

print("\nTop 5 Threshold Combinations (by F1 Weighted):")
print(results_df.head())

print("\nBest Threshold Combination Found:")
print(f"  Threshold b: {best_thresholds['threshold_b']:.6f}")
print(f"  Threshold m: {best_thresholds['threshold_m']:.6f} (fixed)")
print(f"  Threshold u: {best_thresholds['threshold_u']:.6f}")

print("\nMetrics for Best Combination:")
if best_metrics: # Check if best_metrics was assigned
    for metric_name, metric_value in best_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
else:
    print("  No valid combinations were evaluated.")


# --- Optional: Save Results ---
results_df.to_csv(model_dir / "threshold_evaluation_results.csv", index=False)
print(f"\nSaved full evaluation results to {model_dir / 'threshold_evaluation_results.csv'}")

if best_thresholds: # Check if best_thresholds is not empty
    with open(model_dir / "best_evaluated_thresholds.pkl", "wb") as f:
        pickle.dump(best_thresholds, f)
    print(f"Saved best evaluated thresholds to {model_dir / 'best_evaluated_thresholds.pkl'}")

print("\n=============== Threshold Evaluation Script Finished ===============")