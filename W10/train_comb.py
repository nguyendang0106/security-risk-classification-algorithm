import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import pickle
from collections import Counter # Import Counter

# Set fixed seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Add project root to path to allow finding W9
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from sklearn
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

# Import from skopt for Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Import from scipy for hyperparameter distributions
from scipy.stats import randint, loguniform

import W9.util.common as util

print("=============== Combined 2017 + 2018 Dataset Training (Mimicking train2017 Sampling) ===============") # Updated Title

# --- Add verbose flag ---
verbose = True
# --- End Add verbose flag ---

# Create output directory
output_dir = pathlib.Path("W10/models_combined_sampled") # Changed output dir name
output_dir.mkdir(parents=True, exist_ok=True)

# ============================ 1. LOAD AND PREPARE DATA ============================

print("\n=============== Loading data from both 2017 and 2018 datasets ===============")

# Initialize data structures
train = {"ocsvm": {}, "ae": {}, "stage2": {}}
val = {"ocsvm": {}, "ae": {}, "stage2": {}}
test = {}

# Load 2017 data (Load ALL data)
print("\nLoading FULL 2017 dataset...")
data2017_dir = "W9/data1/clean/"
benign2017 = pd.read_parquet(f"{data2017_dir}/all_benign.parquet")
malicious2017 = pd.read_parquet(f"{data2017_dir}/all_malicious.parquet")
print(f"2017 data loaded: {benign2017.shape[0]} benign, {malicious2017.shape[0]} malicious")

# Load 2018 data (Load ALL data)
print("\nLoading FULL 2018 dataset...")
data2018_dir = "W9/data2/clean/"
benign2018 = pd.read_parquet(f"{data2018_dir}/all_benign.parquet")
malicious2018 = pd.read_parquet(f"{data2018_dir}/all_malicious.parquet")
print(f"2018 data loaded: {benign2018.shape[0]} benign, {malicious2018.shape[0]} malicious")

# Add source year column to track origin
benign2017['year'] = 2017
malicious2017['year'] = 2017
benign2018['year'] = 2018
malicious2018['year'] = 2018

# Combine ALL benign data
all_benign = pd.concat([benign2017, benign2018], axis=0, ignore_index=True)
print(f"Combined benign data: {all_benign.shape[0]} samples")
total_benign_original = all_benign.shape[0] # Store original count for overview

# Combine ALL malicious data
all_malicious = pd.concat([malicious2017, malicious2018], axis=0, ignore_index=True)
print(f"Combined malicious data: {all_malicious.shape[0]} samples")
total_malicious_original = all_malicious.shape[0] # Store original count

# --- Capture original attack types BEFORE mapping ---
# Ensure 'Label' exists and handle potential missing values if necessary
if 'Label' not in all_malicious.columns:
    raise ValueError("Column 'Label' not found in malicious data.")
all_malicious['Label'] = all_malicious['Label'].fillna('Unknown_Original') # Or handle differently
original_attack_type = all_malicious['Label'].astype(str).copy() # Ensure string type
# --- End Capture ---

# Standardize labels across datasets
label_mapping = {
    # 2017 dataset
    'DoS Hulk':'(D)DOS',
    'PortScan':'Port Scan',
    'DDoS':'(D)DOS',
    'DoS slowloris':'(D)DOS',
    'DoS Slowhttptest':'(D)DOS',
    'DoS GoldenEye':'(D)DOS',
    'SSH-Patator':'Brute Force',
    'FTP-Patator':'Brute Force',
    'Bot': 'Botnet',
    'Web Attack \x96 Brute Force': 'Web Attack',
    'Web Attack \x96 Sql Injection': 'Web Attack',
    'Web Attack \x96 XSS': 'Web Attack',
    'Infiltration': 'Unknown',
    'Heartbleed': 'Unknown',
    'Web Attack ï¿½ Brute Force': 'Web Attack',
    'Web Attack ï¿½ Sql Injection': 'Web Attack',
    'Web Attack ï¿½ XSS': 'Web Attack',

    # 2018 dataset
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
    'Infilteration': 'Unknown',

    # Handle potential fillna value if needed
    'Unknown_Original': 'Unknown'
}

# Apply label mapping to malicious data
# Create a temporary Series for mapped labels
mapped_labels_temp = original_attack_type.map(lambda x: label_mapping.get(x, 'Unknown'))
# Assign the mapped labels back to the DataFrame
all_malicious['Label'] = mapped_labels_temp
print("\nClass distribution after mapping:")
print(all_malicious['Label'].value_counts())

# --- Malicious Sampling (Mimicking train2017_2.py) ---
MALICIOUS_SAMPLE_SIZE_PER_TYPE = 1948 # As used in train2017_2.py's load_data
print(f"\nSampling up to {MALICIOUS_SAMPLE_SIZE_PER_TYPE} instances per ORIGINAL malicious attack type...")

# Add original attack type back temporarily for sampling
all_malicious['Original_Label'] = original_attack_type

malicious_sampled_list = []
for name, group in all_malicious.groupby('Original_Label'):
    if len(group) > MALICIOUS_SAMPLE_SIZE_PER_TYPE:
        malicious_sampled_list.append(group.sample(n=MALICIOUS_SAMPLE_SIZE_PER_TYPE, random_state=seed_value))
    else:
        malicious_sampled_list.append(group)

malicious_sampled = pd.concat(malicious_sampled_list, ignore_index=False) # Keep original index temporarily
# Keep the Original_Label column, do not drop it here
# Shuffle the sampled data AND reset index
malicious_sampled = malicious_sampled.sample(frac=1, random_state=seed_value).reset_index(drop=True)

print(f"Total malicious samples after sampling: {malicious_sampled.shape[0]}")
print("Sampled malicious class distribution (mapped labels):")
print(malicious_sampled['Label'].value_counts())
total_malicious_sampled = malicious_sampled.shape[0]
# --- End Malicious Sampling ---


# Common preprocessing: drop unnecessary columns
columns_to_drop = ['Timestamp', 'Destination Port']
# Drop Label from benign if it exists (it shouldn't, but safe check)
if 'Label' in all_benign.columns:
    x_benign = all_benign.drop(columns=columns_to_drop + ['Label'], errors='ignore')
else:
    x_benign = all_benign.drop(columns=columns_to_drop, errors='ignore')

# Use the SAMPLED malicious data now
# The 'Original_Label' column is now correctly present in malicious_sampled after shuffling/resetting index
# Remove the incorrect alignment line:
# malicious_sampled['Original_Label'] = original_attack_type[malicious_sampled.index] # REMOVE THIS LINE

# Define feature columns (excluding labels and year)
feature_cols = [col for col in malicious_sampled.columns if col not in columns_to_drop + ['Label', 'Original_Label', 'year']]
x_malicious = malicious_sampled[feature_cols] # Select only feature columns
y_malicious = malicious_sampled['Label'] # Mapped labels from the sampled data
y_malicious_original = malicious_sampled['Original_Label'] # Original labels from the sampled data
malicious_years = malicious_sampled['year'] # Keep track of years for sampled data

# ============================ 2. DATA SPLITTING (Using Sampled Malicious) ============================

print("\n=============== Splitting data (Using Sampled Malicious) ===============")

# Define split proportions
train_prop = 0.70
val_prop = 0.15
test_prop = 0.15

# Split benign data into train/val/test (stratified by year) - Uses ALL benign data
print(f"Splitting ALL benign data ({x_benign.shape[0]} samples)...")
x_benign_train, x_benign_valtest, y_benign_train_year, y_benign_valtest_year = train_test_split(
    x_benign,
    all_benign['year'], # Use year from the original all_benign
    train_size=train_prop,
    random_state=seed_value,
    stratify=all_benign['year']
)
# Adjust validation size relative to the remaining data
val_size_benign_adj = val_prop / (val_prop + test_prop)
x_benign_val, x_benign_test, y_benign_val_year, y_benign_test_year = train_test_split(
    x_benign_valtest,
    y_benign_valtest_year,
    train_size=val_size_benign_adj,
    random_state=seed_value,
    stratify=y_benign_valtest_year
)
print(f"Benign splits: Train={x_benign_train.shape[0]}, Val={x_benign_val.shape[0]}, Test={x_benign_test.shape[0]}")


# Split SAMPLED malicious data into train/val/test (stratified by ORIGINAL label)
print(f"Splitting SAMPLED malicious data ({x_malicious.shape[0]} samples)...")
# Simplify stratification: Use only Original_Label which was used for sampling
# This reduces the chance of creating groups with only 1 member after the first split.

# First split: Separate 70% for training
x_malicious_train, x_malicious_valtest, \
y_malicious_train, y_malicious_valtest, \
y_malicious_original_train, y_malicious_original_valtest, \
year_malicious_train, year_malicious_valtest = train_test_split(
    x_malicious,
    y_malicious, # Mapped labels from sampled data
    y_malicious_original, # Original labels from sampled data
    malicious_years, # Years from sampled data
    train_size=train_prop,
    random_state=seed_value,
    stratify=y_malicious_original # Stratify by Original Label ONLY
)

# Second split: Split the remaining 30% into 15% validation and 15% test (50/50 split of remainder)
# Ensure the stratification target for the second split aligns with the data being split
# Adjust validation size relative to the remaining data is incorrect here, should be 50% of remainder
test_size_malicious_adj = test_prop / (val_prop + test_prop) # This calculates to 0.5

x_malicious_val, x_malicious_test, \
y_malicious_val, y_malicious_test, \
y_malicious_original_val, y_malicious_original_test, \
year_malicious_val, year_malicious_test = train_test_split(
    x_malicious_valtest,
    y_malicious_valtest, # Mapped labels
    y_malicious_original_valtest, # Original labels
    year_malicious_valtest, # Years
    test_size=test_size_malicious_adj, # Split remainder 50/50
    random_state=seed_value,
    stratify=y_malicious_original_valtest # Stratify the second split by Original Label ONLY
)
print(f"Sampled Malicious splits: Train={x_malicious_train.shape[0]}, Val={x_malicious_val.shape[0]}, Test={x_malicious_test.shape[0]}")


# Save the year information for the FINAL test set (for later evaluation by year)
# This test set combines the benign test split and the sampled malicious test split
test_years = {
    'benign': y_benign_test_year.values,
    'malicious': year_malicious_test.values # Use the year info from the malicious test split
}
with open(output_dir / "test_years.pkl", "wb") as f:
    pickle.dump(test_years, f)
print(f"Saved test year info to {output_dir / 'test_years.pkl'}")

# Remove year column before training (if it still exists after potential drops)
# This should be done *after* splitting and saving year info
columns_to_drop_after_split = ['year']
if 'year' in x_benign_train.columns:
    x_benign_train = x_benign_train.drop(columns=columns_to_drop_after_split)
if 'year' in x_benign_val.columns:
    x_benign_val = x_benign_val.drop(columns=columns_to_drop_after_split)
if 'year' in x_benign_test.columns:
    x_benign_test = x_benign_test.drop(columns=columns_to_drop_after_split)
if 'year' in x_malicious_train.columns:
    x_malicious_train = x_malicious_train.drop(columns=columns_to_drop_after_split)
if 'year' in x_malicious_val.columns:
    x_malicious_val = x_malicious_val.drop(columns=columns_to_drop_after_split)
if 'year' in x_malicious_test.columns:
    x_malicious_test = x_malicious_test.drop(columns=columns_to_drop_after_split)
print("Removed 'year' column from feature sets.")

# --- Data Overview Printout (Adjusted for Sampling and Original Labels) ---
if verbose:
    print("\n--- Data Split Overview (Post Sampling & Splitting) ---")
    overview = {}
    # Benign overview (remains the same)
    overview[('Benign', 'Benign')] = {
        "#Original": total_benign_original,
        "#Sampled/SplitTotal": x_benign_train.shape[0] + x_benign_val.shape[0] + x_benign_test.shape[0],
        "#Train": x_benign_train.shape[0],
        "#Validation": x_benign_val.shape[0],
        "#Test": x_benign_test.shape[0],
    }

    # Malicious overview (detailed by original label)
    # Use Counter for efficient counting on original labels in splits
    train_orig_counts = Counter(y_malicious_original_train)
    val_orig_counts = Counter(y_malicious_original_val)
    test_orig_counts = Counter(y_malicious_original_test)

    # Get unique mapped labels from the sampled set
    all_sampled_mapped_labels = sorted(list(y_malicious.unique()))

    # Create a mapping from original label to mapped label for easier lookup
    original_to_mapped = all_malicious.set_index('Original_Label')['Label'].to_dict()

    # Get unique original labels present in the sampled data
    unique_original_labels_sampled = sorted(list(y_malicious_original.unique()))

    for mapped_class in all_sampled_mapped_labels:
        # Find original labels that map to this mapped_class AND are in the sampled data
        original_labels_in_class = [
            orig_label for orig_label in unique_original_labels_sampled
            if original_to_mapped.get(orig_label) == mapped_class
        ]

        class_train_total = 0
        class_val_total = 0
        class_test_total = 0
        class_sampled_total = 0
        class_original_total = 0

        for original_label in sorted(original_labels_in_class):
            train_count = train_orig_counts.get(original_label, 0)
            val_count = val_orig_counts.get(original_label, 0)
            test_count = test_orig_counts.get(original_label, 0)

            # Count in original combined malicious data (before sampling)
            original_total_for_impl = (original_attack_type == original_label).sum()
            # Count in the sampled malicious data
            sampled_total_for_impl = (y_malicious_original == original_label).sum()

            overview[(mapped_class, original_label)] = {
                "#Original": original_total_for_impl,
                "#Sampled/SplitTotal": sampled_total_for_impl,
                "#Train": train_count,
                "#Validation": val_count,
                "#Test": test_count,
            }
            class_train_total += train_count
            class_val_total += val_count
            class_test_total += test_count
            class_sampled_total += sampled_total_for_impl
            class_original_total += original_total_for_impl

        # Add the 'ALL' summary row for the mapped class
        overview[(mapped_class, 'ALL')] = {
            "#Original": class_original_total,
            "#Sampled/SplitTotal": class_sampled_total,
            "#Train": class_train_total,
            "#Validation": class_val_total,
            "#Test": class_test_total,
        }

    # Displaying the overview DataFrame
    overview_df = pd.DataFrame.from_dict(overview, orient="index")
    # Ensure column order is correct
    overview_df = overview_df[["#Original", "#Sampled/SplitTotal", "#Train", "#Validation", "#Test"]]
    print(overview_df.rename_axis(["Class", "Impl"]))

# --- End Data Overview Printout ---

# Prepare data for anomaly detection (OCSVM / AE)
print("\nPreparing data for Stage 1 (Anomaly Detection)")

# --- Sample benign training data ---
# Use the BENIGN training split (70% of all benign)
n_ae_train = 100000
n_ocsvm_train = 10000

print(f"Sampling {n_ae_train} benign samples from benign training split for main Stage 1 training (AE/OCSVM)...")
train_sample_100k = x_benign_train.sample(n=min(n_ae_train, x_benign_train.shape[0]), random_state=seed_value)
train["ae"]["x"] = train_sample_100k
train["ae"]["y"] = np.ones(train["ae"]["x"].shape[0]) # Labels for the 100k sample

print(f"Sampling {n_ocsvm_train} benign samples from benign training split for comparison Stage 1 training (OCSVM)...")
train_sample_10k = x_benign_train.sample(n=min(n_ocsvm_train, x_benign_train.shape[0]), random_state=seed_value)
train["ocsvm"]["x"] = train_sample_10k
train["ocsvm"]["y"] = np.ones(train["ocsvm"]["x"].shape[0]) # Labels for the 10k sample
# --- End Sampling ---

# Prepare validation data (mixed benign and malicious)
# Uses the BENIGN validation split (15% of all benign)
# Uses the SAMPLED MALICIOUS validation split (15% of sampled malicious)
val["ocsvm"]["x"] = pd.concat([x_benign_val, x_malicious_val], axis=0, ignore_index=True)
val["ocsvm"]["y"] = np.concatenate([np.ones(x_benign_val.shape[0]), np.full(x_malicious_val.shape[0], -1)])

val["ae"]["x"] = val["ocsvm"]["x"].copy()
val["ae"]["y"] = val["ocsvm"]["y"].copy()

# Prepare data for Stage 2 (Classification)
# Uses the SAMPLED MALICIOUS training split (70% of sampled malicious)
print("Preparing data for Stage 2 (Classification)")
train["stage2"]["x"] = x_malicious_train
train["stage2"]["y"] = y_malicious_train # Mapped labels for training RF

# Validation set for Stage 2: Use 1500 benign samples from the BENIGN validation split + SAMPLED MALICIOUS validation split
n_benign_val_stage2 = 1500
# Sample from the BENIGN validation split
benign_val_stage2 = x_benign_val.sample(n=min(n_benign_val_stage2, x_benign_val.shape[0]), random_state=seed_value)

val["stage2"]["x"] = pd.concat([benign_val_stage2, x_malicious_val], axis=0, ignore_index=True)
# Labels: "Unknown" for the benign samples, actual mapped labels for the malicious validation samples
val["stage2"]["y"] = np.concatenate([np.full(benign_val_stage2.shape[0], "Unknown"), y_malicious_val])

# One-hot encode stage2 labels
train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

# Prepare FINAL test data
# Uses the BENIGN test split (15% of all benign)
# Uses the SAMPLED MALICIOUS test split (15% of sampled malicious)
test['x'] = pd.concat([x_benign_test, x_malicious_test], axis=0, ignore_index=True)
# Ground truth labels for the final test set
test["y"] = np.concatenate([np.full(x_benign_test.shape[0], "Benign"), y_malicious_test]) # Mapped labels for test ground truth
# Binary labels for OCSVM testing
test["y_n"] = np.concatenate([np.ones(x_benign_test.shape[0]), np.full(x_malicious_test.shape[0], -1)])
# Final ground truth labels (handling 'Unknown' mapping)
test["y_unknown"] = np.where(test["y"] == "Unknown", "Unknown", test["y"])
# Alternative ground truth where Benign is also mapped to Unknown (as in original script)
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])

print(f"\nFinal Data Shapes Prepared:")
print(f"  Train OCSVM (Benign): {train['ocsvm']['x'].shape}")
print(f"  Train AE (Benign): {train['ae']['x'].shape}")
print(f"  Train Stage2 (Malicious): {train['stage2']['x'].shape}")
print(f"  Val OCSVM/AE (Mixed): {val['ocsvm']['x'].shape}")
print(f"  Val Stage2 (Mixed): {val['stage2']['x'].shape}")
print(f"  Test (Mixed): {test['x'].shape}")


# ============================ 3. FEATURE SELECTION ============================
# This section remains largely the same, but trains the temp RF on the new train['stage2']['x']

print("\n=============== Performing Feature Selection ===============")

N_FEATURES_TO_SELECT = 15

# Train a temporary RF for feature selection based on Stage 2 data (now the sampled malicious train split)
print("Training temporary RF for feature selection...")
# Ensure train['stage2']['x'] is not empty
if train['stage2']['x'].empty:
    raise ValueError("Stage 2 training data (train['stage2']['x']) is empty. Check sampling/splitting.")

temp_rf = RandomForestClassifier(n_estimators=20, random_state=seed_value, n_jobs=-1)
temp_rf.fit(train['stage2']['x'], train['stage2']['y'])

# Get feature importances and select top N
importances = temp_rf.feature_importances_
indices = np.argsort(importances)[::-1]
selected_indices = indices[:N_FEATURES_TO_SELECT]

# Print selected feature names
# Ensure feature names are available (should be if DataFrames were used)
if isinstance(train['stage2']['x'], pd.DataFrame):
    feature_names = train['stage2']['x'].columns.values
    selected_features = feature_names[selected_indices]
    print(f"Selected top {N_FEATURES_TO_SELECT} features:")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1}. {feature}: {importances[indices[i]]:.4f}")
else:
    selected_features = [f"Feature_{i}" for i in selected_indices] # Placeholder if no names
    print(f"Selected top {N_FEATURES_TO_SELECT} feature indices: {selected_indices}")


# Save selected feature indices
with open(output_dir / "selected_feature_indices.pkl", "wb") as f:
    pickle.dump(selected_indices, f)
print(f"Saved selected feature indices to {output_dir / 'selected_feature_indices.pkl'}")

# Apply feature selection to all datasets
print("Applying feature selection to all data splits...")

# Helper function to apply feature selection
def select_features(df, indices_to_keep):
    if isinstance(df, pd.DataFrame):
        return df.iloc[:, indices_to_keep]
    elif isinstance(df, np.ndarray):
        return df[:, indices_to_keep]
    else:
        # Handle potential None or other types if necessary
        if df is None:
            return None
        raise TypeError(f"Unexpected data type for feature selection: {type(df)}")

# Apply to training and validation sets
for data_dict in [train, val]:
    for key in data_dict:
        if 'x' in data_dict[key] and data_dict[key]['x'] is not None:
             # Check shape before selecting
            if data_dict[key]['x'].shape[1] > max(selected_indices):
                data_dict[key]['x'] = select_features(data_dict[key]['x'], selected_indices)
            else:
                print(f"Warning: Skipping feature selection for data_dict[{key}]['x'] due to shape mismatch.")


# Apply to test data
if 'x' in test and test['x'] is not None:
     # Check shape before selecting
    if test['x'].shape[1] > max(selected_indices):
        test['x'] = select_features(test['x'], selected_indices)
    else:
        print(f"Warning: Skipping feature selection for test['x'] due to shape mismatch.")


print(f"Feature selection applied. New feature count (test set): {test['x'].shape[1] if test.get('x') is not None else 'N/A'}")


# ============================ 4. SCALING DATA ============================
# This section remains the same, fitting/transforming the feature-selected data

print("\n=============== Scaling Selected Features ===============")

# Stage 1 (OCSVM) scaling - Fit on 10k benign sample
scaler_ocsvm = QuantileTransformer(output_distribution='normal')
train['ocsvm']['x_s'] = scaler_ocsvm.fit_transform(train['ocsvm']['x'])
val['ocsvm']['x_s'] = scaler_ocsvm.transform(val['ocsvm']['x'])
test['ocsvm_s'] = scaler_ocsvm.transform(test['x'])

# Stage 1 (AE) scaling - Fit on 100k benign sample
scaler_ae = QuantileTransformer(output_distribution='normal')
train['ae']['x_s'] = scaler_ae.fit_transform(train['ae']['x'])
val['ae']['x_s'] = scaler_ae.transform(val['ae']['x'])
test['ae_s'] = scaler_ae.transform(test['x'])

# Save Stage 1 scaler (using the one fit on 100k as primary)
with open(output_dir / "stage1_ocsvm_scaler.p", "wb") as f:
    pickle.dump(scaler_ae, f)
print(f"Saved stage1_ocsvm_scaler.p to {output_dir}")

# Stage 2 (RF) scaling - Fit on the SAMPLED malicious training data
scaler_stage2 = QuantileTransformer(output_distribution='normal')
train['stage2']['x_s'] = scaler_stage2.fit_transform(train['stage2']['x'])
val['stage2']['x_s'] = scaler_stage2.transform(val['stage2']['x'])
test['stage2_s'] = scaler_stage2.transform(test['x'])

# Save Stage 2 scaler
with open(output_dir / "stage2_rf_scaler.p", "wb") as f:
    pickle.dump(scaler_stage2, f)
print(f"Saved stage2_rf_scaler.p to {output_dir}")
with open(output_dir / "baseline_rf_scaler.p", "wb") as f:
    pickle.dump(scaler_stage2, f)
print(f"Saved baseline_rf_scaler.p to {output_dir}")

# Additional uniform scaling (if needed later, applied to same data)
scaler_uniform = QuantileTransformer(output_distribution='uniform')
train['stage2']['x_q'] = scaler_uniform.fit_transform(train['stage2']['x'])
val['stage2']['x_q'] = scaler_uniform.transform(val['stage2']['x'])
test['stage2_q'] = scaler_uniform.transform(test['x'])

# ============================ 5. STAGE 1: ONE-CLASS SVM ============================
# This section remains the same - trains on the 10k/100k sampled benign data

print("\n=============== Training Stage 1: One-Class SVM ===============")

ocsvm_pipeline_base = Pipeline(
    [
        ("imputer", SimpleImputer(strategy='mean')),
        ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)),
        ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)) # Set verbose=False
    ]
)

# ocsvm_param_dist = {
#     'pca__n_components': Integer(10, 15),
#     'ocsvm__gamma': Real(1e-2, 1e-1, prior='log-uniform'),
#     'ocsvm__nu': Real(1e-4, 1e-3, prior='log-uniform'),
#     'ocsvm__kernel': Categorical(['rbf'])
# }

ocsvm_param_dist = {
    'pca__n_components': randint(10, 15), # Sample integers from 10 up to (but not including) 16
    'ocsvm__gamma': loguniform(1e-5, 1e0), # Sample from log-uniform distribution between 
    'ocsvm__nu': loguniform(1e-5, 1e0), # Sample from log-uniform distribution between 
    'ocsvm__kernel': ['rbf'] # Keep kernel fixed or add others like 'poly' if desired
}

print("\n--- Optimizing Stage 1 OCSVM using BayesSearchCV ---")
n_iterations = 15 # Reduced iterations for potentially faster run, adjust if needed
clf_ocsvm = RandomizedSearchCV(
    estimator=ocsvm_pipeline_base,
    param_distributions=ocsvm_param_dist,
    n_iter=n_iterations,
    scoring='roc_auc', # Use AUC for optimization
    cv=5, # Reduced CV folds for speed
    verbose=2, # Reduced verbosity
    n_jobs=-1,
    random_state=seed_value
)

# Train on 10k benign samples
# print(f"\nStarting BayesSearchCV fit ({n_iterations} iterations) on {train['ocsvm']['x_s'].shape[0]} benign samples (10k set)...")
# clf_ocsvm.fit(train['ocsvm']['x_s'], train['ocsvm']['y'])
# print("\nBayesSearchCV for OCSVM (10k) finished.")
# print("Best ROC AUC score (10k):", clf_ocsvm.best_score_)
# print("Best parameters (10k):", dict(clf_ocsvm.best_params_))
# ocsvm_model_10k = clf_ocsvm.best_estimator_
# with open(output_dir / "stage1_ocsvm_model_10k.p", "wb") as f:
#     pickle.dump(ocsvm_model_10k, f)
# print(f"Saved stage1_ocsvm_model_10k.p to {output_dir}")
f = open("W10/models_combined_sampled/stage1_ocsvm_model_10k.p", "rb")
ocsvm_model_10k = pickle.load(f)
f.close()

# Train on 100k benign samples (using the same search instance, refit)
# print(f"\nStarting BayesSearchCV fit ({n_iterations} iterations) on {train['ae']['x_s'].shape[0]} benign samples (100k set)...")
# # It's generally better to use a fresh BayesSearchCV instance if resources allow,
# # but refitting is faster here. Results might differ slightly from independent runs.
# clf_ocsvm.fit(train['ae']['x_s'], train['ae']['y'])
# print("\nBayesSearchCV for OCSVM (100k) finished.")
# print("Best ROC AUC score (100k):", clf_ocsvm.best_score_)
# print("Best parameters (100k):", dict(clf_ocsvm.best_params_))
# ocsvm_model_100k = clf_ocsvm.best_estimator_

# with open(output_dir / "stage1_ocsvm.p", "wb") as f:
#     pickle.dump(ocsvm_model_100k, f)
# print(f"Saved stage1_ocsvm.p to {output_dir}")
# with open(output_dir / "stage1_ocsvm_model.p", "wb") as f:
#     pickle.dump(ocsvm_model_100k.named_steps['ocsvm'], f)
# print(f"Saved stage1_ocsvm_model.p to {output_dir}")
f = open("W10/models_combined_sampled/stage1_ocsvm.p", "rb")
ocsvm_model_100k = pickle.load(f)
f.close()


# Validation
print("\nValidating OCSVM model (10k)...")
score_val_10k = -ocsvm_model_10k.decision_function(val['ocsvm']['x_s'])
curves_metrics_10k, summary_metrics_10k = util.evaluate_proba(val['ocsvm']['y'], score_val_10k)
print("OCSVM (10k) Validation Summary (Top F1):")
print(summary_metrics_10k[summary_metrics_10k['metric'] == 'F1'])
print(summary_metrics_10k)


print("\nValidating OCSVM model (100k)...")
score_val_100k = -ocsvm_model_100k.decision_function(val['ae']['x_s'])
curves_metrics_100k, summary_metrics_100k = util.evaluate_proba(val['ae']['y'], score_val_100k)
print("OCSVM (100k) Validation Summary (Top F1):")
print(summary_metrics_100k[summary_metrics_100k['metric'] == 'F1'])
print(summary_metrics_100k)



# Define Thresholds based on 100k model validation (as it's the primary one used later)
print("\nThresholds based on 100k model validation:")
f1_threshold_100k = summary_metrics_100k.loc[summary_metrics_100k.metric == 'F8', 'threshold'].iloc[0]
print(f"  F1 Threshold: {f1_threshold_100k}")
# Calculate quantiles on benign validation scores for potential use
quantiles_list = [0.995, 0.99, 0.95, 0.9, 0.875, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
benign_scores_val_100k = score_val_100k[val["ae"]["y"] == 1]
quantile_thresholds_100k = {q: np.quantile(benign_scores_val_100k, q) for q in quantiles_list}
print("  Quantile thresholds (benign scores):")
for q, t in quantile_thresholds_100k.items():
    print(f"    {q:.2f}: {t:.6f}")

#  Define Thresholds
quantiles = [0.995, 0.99, 0.95, 0.9, 0.875, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
print("Thresholds based on 10k model validation:")
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics_10k.metric, summary_metrics_10k.FPR, summary_metrics_10k.threshold)})
print({q: np.quantile(score_val_10k[val["ocsvm"]["y"] == 1], q) for q in quantiles})

quantiles = [0.995, 0.99, 0.95, 0.9, 0.875, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
print("\nThresholds based on 100k model validation:")
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics_100k.metric, summary_metrics_100k.FPR, summary_metrics_100k.threshold)})
print({q: np.quantile(score_val_100k[val["ae"]["y"] == 1], q) for q in quantiles})


# Test
print("\nTesting OCSVM model (10k)...")
score_test_10k = -ocsvm_model_10k.decision_function(test['ocsvm_s'])
curves_metrics_test_10k, summary_metrics_test_10k = util.evaluate_proba(test["y_n"], score_test_10k)
print("OCSVM (10k) Test Summary (Top F1):")
print(summary_metrics_test_10k[summary_metrics_test_10k['metric'] == 'F1'])
print(summary_metrics_test_10k)


print("\nTesting OCSVM model (100k)...")
score_test_100k = -ocsvm_model_100k.decision_function(test['ae_s']) # Use the scores from the 100k model for the pipeline
curves_metrics_test_100k, summary_metrics_test_100k = util.evaluate_proba(test["y_n"], score_test_100k)
print("OCSVM (100k) Test Summary (Top F1):")
print(summary_metrics_test_100k[summary_metrics_test_100k['metric'] == 'F1'])
print(summary_metrics_test_100k)


# ============================ 6. STAGE 2: RANDOM FOREST ============================
# This section remains the same - trains on the SAMPLED malicious training data

print("\n=============== Training Stage 2: Random Forest ===============")

# Calculate class weights for balanced training (using the new train["stage2"]["y"])
print("Calculating class weights for Stage 2...")
unique_classes_stage2 = np.unique(train["stage2"]["y"])
# Handle case where a class might be missing after sampling/splitting
if len(unique_classes_stage2) > 1:
    class_weights_array_stage2 = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_stage2,
        y=train["stage2"]["y"]
    )
    class_weights_dict_stage2 = dict(zip(unique_classes_stage2, class_weights_array_stage2))
    print("Stage 2 Class weights:", class_weights_dict_stage2)
else:
    print("Warning: Only one class present in Stage 2 training data. Using uniform weights.")
    class_weights_dict_stage2 = None # Or 'balanced' if RF handles it

# Train baseline random forest (without anomaly score)
print("\nTraining Stage 2 baseline Random Forest (without anomaly score)...")
# Define base RF model
rf_baseline_base = RandomForestClassifier(
    n_estimators=70,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=class_weights_dict_stage2,
    ccp_alpha=0.0,
    max_samples=None
)

# Hyperparameter grid for GridSearchCV
hyperparameters_rf = {
    'n_estimators': [50, 70, 75, 90, 100, 125, 150] # Reduced grid for speed
}

# Use GridSearchCV to find best n_estimators for baseline
clf_rf_baseline = GridSearchCV(
    estimator=rf_baseline_base,
    param_grid=hyperparameters_rf,
    cv=5, # Reduced CV folds
    verbose=1,
    n_jobs=-1
)

# clf_rf_baseline.fit(train['stage2']['x_s'], train["stage2"]["y"])

# print("\nGridSearchCV for Baseline RF finished.")
# print("---------------")
# print("Best score on Validation set:", clf_rf_baseline.best_score_)
# print("---------------")
# print("Best parameters:", clf_rf_baseline.best_params_)
# print("---------------")
# print("Best estimator:", clf_rf_baseline.best_estimator_)
# print("---------------")

# # Use the best model found by GridSearchCV
# rf_model_baseline = clf_rf_baseline.best_estimator_
# print("\nUsing best estimator from GridSearchCV for Stage 2 Random Forest (Baseline)...")

# # Save the baseline model
# with open(output_dir / "baseline_rf.p", "wb") as f:
#     pickle.dump(rf_model_baseline, f)
# print(f"Saved baseline_rf.p to {output_dir}")
f = open("W10/models_combined_sampled/baseline_rf.p", "rb")
rf_model_baseline = pickle.load(f)
f.close()

# Validate the baseline model
print("\nValidating Stage 2 Random Forest (Baseline)...")
y_proba_val_baseline = rf_model_baseline.predict_proba(val['stage2']['x_s'])
# Find optimal threshold based on validation F1 weighted score
fmacro_bl, fweight_bl, threshold_fscore_bl, f_best_bl = util.optimal_fscore_multi(
    val['stage2']['y'], y_proba_val_baseline, rf_model_baseline.classes_
)
optimal_threshold_m_baseline = f_best_bl["f1_weighted_threshold"]
print(f"Optimal F1 weighted threshold (Baseline Validation): {optimal_threshold_m_baseline:.4f}")

# Test baseline model using the optimal threshold found during validation
print("\nTesting Stage 2 Random Forest (Baseline)...")
y_proba_test_baseline = rf_model_baseline.predict_proba(test['stage2_s'])
y_pred_test_baseline = np.where(
    np.max(y_proba_test_baseline, axis=1) > optimal_threshold_m_baseline, # Use optimal threshold
    rf_model_baseline.classes_[np.argmax(y_proba_test_baseline, axis=1)],
    'Unknown'
)
print("Stage 2 Baseline Test Metrics (with optimal thresholding):")
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_baseline, average='macro', zero_division=0),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_baseline, average='weighted', zero_division=0),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_baseline),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_baseline)
})


# --- Train Stage 2 RF with Anomaly Score ---
print("\nPreparing data for Stage 2 RF with anomaly score...")
# Use the primary 100k OCSVM model to generate scores
proba_train_stage2 = -ocsvm_model_100k.decision_function(train['stage2']['x_s']) # Scores for malicious train
proba_val_stage2 = -ocsvm_model_100k.decision_function(val['stage2']['x_s'])     # Scores for stage 2 val set
proba_test_stage2 = -ocsvm_model_100k.decision_function(test['stage2_s'])      # Scores for final test set

# Stack features with anomaly scores
train_with_proba = np.column_stack((train['stage2']['x_s'], proba_train_stage2))
val_with_proba = np.column_stack((val['stage2']['x_s'], proba_val_stage2))
test_with_proba = np.column_stack((test['stage2_s'], proba_test_stage2))
print("Shapes with anomaly score:", train_with_proba.shape, val_with_proba.shape, test_with_proba.shape)

print("\nTraining Stage 2 Random Forest (with extra feature)...")
# Use the same base RF model structure but train on data with the extra feature
# Use the best n_estimators found for the baseline model
# rf_model_extra_feature = RandomForestClassifier(
#     n_estimators=clf_rf_baseline.best_params_['n_estimators'], # Use 125
#     class_weight=class_weights_dict_stage2,       # Include class weights
#     random_state=seed_value,                      # Set for reproducibility
#     # Keep other parameters like max_features='sqrt' as used in the grid search default
#     max_features='sqrt'
#     # Add other parameters if you specifically tuned them and want to keep them
# )

# rf_model_extra_feature.fit(train_with_proba, train["stage2"]["y"])
# print("Stage 2 Random Forest (with extra feature) trained.")

# # Save the enhanced model
# with open(output_dir / "stage2_rf_model.p", "wb") as f:
#     pickle.dump(rf_model_extra_feature, f)
# print(f"Saved stage2_rf_model.p to {output_dir}")
# with open(output_dir / "stage2_rf.p", "wb") as f: # Keep both names for compatibility
#     pickle.dump(rf_model_extra_feature, f)
# print(f"Saved stage2_rf.p to {output_dir}")
# with open(output_dir / "sota_stage2.p", "wb") as f: # Keep SOTA name for compatibility
#     pickle.dump(rf_model_extra_feature, f)
# print(f"Saved sota_stage2.p to {output_dir}")
f = open("W10/models_combined_sampled/stage2_rf.p", "rb")
rf_model_extra_feature = pickle.load(f)
f.close()

# Validate the enhanced model
print("\nValidating Stage 2 Random Forest (with extra feature)...")
y_proba_val_extra = rf_model_extra_feature.predict_proba(val_with_proba)
# Find optimal threshold based on validation F1 weighted score
fmacro_ext, fweight_ext, threshold_fscore_ext, f_best_ext = util.optimal_fscore_multi(
    val['stage2']['y'], y_proba_val_extra, rf_model_extra_feature.classes_
)
optimal_threshold_m_extra = f_best_ext["f1_weighted_threshold"]
print(f"Optimal F1 weighted threshold (Extra Feature Validation): {optimal_threshold_m_extra:.4f}")

# Test enhanced model using the optimal threshold found during validation
print("\nTesting Stage 2 Random Forest (with extra feature)...")
y_proba_test_extra = rf_model_extra_feature.predict_proba(test_with_proba)
y_pred_test_extra = np.where(
    np.max(y_proba_test_extra, axis=1) > optimal_threshold_m_extra, # Use optimal threshold
    rf_model_extra_feature.classes_[np.argmax(y_proba_test_extra, axis=1)],
    'Unknown'
)
print("Stage 2 Extra Feature Test Metrics (with optimal thresholding):")
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_extra, average='macro', zero_division=0),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_extra, average='weighted', zero_division=0),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_extra),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_extra)
})


# ============================ 7. DETERMINE OPTIMAL THRESHOLDS FOR PIPELINE ============================
# Use thresholds derived from validation steps

print("\n=============== Determining Optimal Thresholds for Pipeline ===============")

# Stage 1 threshold (Benign vs. Fraud) - Use F1 threshold from 100k OCSVM validation
threshold_b = f1_threshold_100k
print(f"Using Stage 1 threshold (threshold_b): {threshold_b:.6f} (from 100k OCSVM F1 validation)")

# Stage 2 threshold (Multi-class classification) - Use threshold from enhanced RF validation
threshold_m = optimal_threshold_m_extra
print(f"Using Stage 2 threshold (threshold_m): {threshold_m:.6f} (from enhanced RF F1 validation)")

# Extension threshold (Unknown handling) - Use a quantile threshold from 100k OCSVM validation
# Let's choose the 90th percentile as a starting point (similar to original script)
threshold_u = quantile_thresholds_100k[0.9]
print(f"Using Extension threshold (threshold_u): {threshold_u:.6f} (90th percentile of benign scores)")

# Store the final thresholds
thresholds = {
    'threshold_b': threshold_b,
    'threshold_m': threshold_m,
    'threshold_u': threshold_u
}
with open(output_dir / "optimal_thresholds.pkl", "wb") as f:
    pickle.dump(thresholds, f)
print(f"Saved optimal thresholds to {output_dir / 'optimal_thresholds.pkl'}")

# Quick check of pipeline performance with these thresholds on the full test set
print("\nQuick check of pipeline performance on full test set using determined thresholds...")
# Use scores from 100k OCSVM test run
y_proba_1_test = score_test_100k
# Use probabilities from enhanced RF test run
y_proba_2_test = y_proba_test_extra

# Apply Stage 1
y_pred_pipeline = np.where(y_proba_1_test < threshold_b, "Benign", "Fraud").astype(object)
# Apply Stage 2
fraud_indices_pipe = np.where(y_pred_pipeline == "Fraud")[0]
if len(fraud_indices_pipe) > 0:
    y_pred_2_pipe = np.where(
        np.max(y_proba_2_test[fraud_indices_pipe], axis=1) > threshold_m,
        rf_model_extra_feature.classes_[np.argmax(y_proba_2_test[fraud_indices_pipe], axis=1)],
        'Unknown'
    )
    y_pred_pipeline[fraud_indices_pipe] = y_pred_2_pipe
# Apply Extension Stage
unknown_indices_pipe = np.where(y_pred_pipeline == "Unknown")[0]
if len(unknown_indices_pipe) > 0:
    y_pred_3_pipe = np.where(y_proba_1_test[unknown_indices_pipe] < threshold_u, "Benign", "Unknown")
    y_pred_pipeline[unknown_indices_pipe] = y_pred_3_pipe

print("Pipeline Quick Check Metrics (Full Test Set):")
print({
    "f1_macro": f1_score(test["y_unknown"], y_pred_pipeline, average='macro', zero_division=0),
    "f1_weighted": f1_score(test["y_unknown"], y_pred_pipeline, average='weighted', zero_division=0),
    'accuracy': accuracy_score(test["y_unknown"], y_pred_pipeline),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown"], y_pred_pipeline)
})
# Plot confusion matrix for this quick check
print("Plotting confusion matrix for quick check...")
all_classes_pipe = sorted(list(np.unique(np.concatenate((test['y_unknown'], y_pred_pipeline)))))
plt.figure(figsize=(10, 8))
util.plot_confusion_matrix(
    test['y_unknown'],
    y_pred_pipeline,
    values=all_classes_pipe,
    labels=all_classes_pipe,
    title="Pipeline Quick Check (Full Test Data)"
)
plt.tight_layout()
plt.show() # Display this one


# ============================ 8. FINAL EVALUATION (Separated by Year) ============================
# This section remains the same - it uses the final models and thresholds
# to evaluate performance on 2017 and 2018 subsets of the test data.

print("\n=============== Final Evaluation (Separated by Year) ===============")

# Evaluate the combined pipeline on the full test set
def evaluate_pipeline(x_test_scaled_ocsvm, x_test_scaled_rf, y_true, ocsvm_model, rf_model, thresholds): # Modified to accept both scaled inputs
    """Evaluates the full 3-stage pipeline."""
    # Stage 1: Anomaly Detection (using OCSVM scaled data)
    score_test = -ocsvm_model.decision_function(x_test_scaled_ocsvm)
    y_pred = np.where(score_test < thresholds['threshold_b'], "Benign", "Fraud").astype(object)

    # Get indices for samples predicted as fraud
    fraud_indices = np.where(y_pred == "Fraud")[0]

    # Stage 2: Classification (only for samples classified as fraud)
    if len(fraud_indices) > 0:
        # Prepare RF input: RF scaled features + anomaly score
        # Select the corresponding RF scaled features and scores
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
        # Update the main prediction array
        y_pred[fraud_indices] = y_pred_2

    # Extension: Reconsider "Unknown" predictions
    unknown_indices = np.where(y_pred == "Unknown")[0]
    if len(unknown_indices) > 0:
        score_test_unknown = score_test[unknown_indices]
        # Apply threshold_u
        y_pred_3 = np.where(score_test_unknown < thresholds['threshold_u'], "Benign", "Unknown")
        # Update the main prediction array
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

# Evaluate on the full test set (using the function for consistency)
print("\nEvaluating full pipeline on entire test set...")
# We need both OCSVM-scaled and RF-scaled test data
y_pred_full, metrics_full = evaluate_pipeline(
    test['ae_s'],             # OCSVM-scaled data (from 100k scaler)
    test['stage2_s'],         # RF-scaled data
    test['y_unknown'],        # Ground truth
    ocsvm_model_100k,         # Trained OCSVM pipeline
    rf_model_extra_feature,   # Trained RF model (with extra feature)
    thresholds                # Determined optimal thresholds
)

print("Full test set metrics (Final Pipeline):")
for metric_name, metric_value in metrics_full.items():
    print(f"  {metric_name}: {metric_value:.4f}")

# Plot confusion matrix for the full test set (Final Pipeline)
all_classes_full = sorted(list(np.unique(np.concatenate((test['y_unknown'], y_pred_full)))))
plt.figure(figsize=(10, 8))
util.plot_confusion_matrix(
    test['y_unknown'],
    y_pred_full,
    values=all_classes_full,
    labels=all_classes_full,
    title="Final Pipeline Performance (All Test Data)"
)
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix_full_final.png")
print(f"Saved final full confusion matrix to {output_dir / 'confusion_matrix_full_final.png'}")
plt.close()

# Separate evaluation for 2017 and 2018 data
print("\n=============== Final Separate Evaluation by Year ===============")

# Load the saved year information
with open(output_dir / "test_years.pkl", "rb") as f:
    test_years_loaded = pickle.load(f)

# Get indices for 2017 and 2018 data from the combined test set
# Ensure the concatenation order matches how test['x'] was created (benign first, then malicious)
indices_2017 = np.where(np.concatenate([test_years_loaded['benign'], test_years_loaded['malicious']]) == 2017)[0]
indices_2018 = np.where(np.concatenate([test_years_loaded['benign'], test_years_loaded['malicious']]) == 2018)[0]

# Define the base classes list, including Port Scan
base_classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']

# Initialize metrics dictionaries in case one year has no data
metrics_2017 = None
metrics_2018 = None

# Evaluate 2017 data
if len(indices_2017) > 0:
    print(f"\nEvaluating on 2017 test set ({len(indices_2017)} samples)...")
    x_test_ocsvm_2017 = test['ae_s'][indices_2017] # OCSVM-scaled
    x_test_rf_2017 = test['stage2_s'][indices_2017] # RF-scaled
    y_true_2017 = test['y_unknown'][indices_2017]

    y_pred_2017, metrics_2017 = evaluate_pipeline(
        x_test_ocsvm_2017, x_test_rf_2017, y_true_2017, ocsvm_model_100k, rf_model_extra_feature, thresholds
    )
    print("2017 test set metrics:")
    for metric_name, metric_value in metrics_2017.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # Plot confusion matrix for 2017
    classes_2017 = sorted(list(np.unique(np.concatenate((y_true_2017, y_pred_2017)))))
    plot_labels_2017 = sorted(list(set(base_classes + classes_2017)))
    plt.figure(figsize=(10, 8))
    util.plot_confusion_matrix(
        y_true_2017,
        y_pred_2017,
        values=plot_labels_2017,
        labels=plot_labels_2017,
        title="Final Pipeline Performance (2017 Test Data)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_2017_final.png")
    print(f"Saved 2017 confusion matrix to {output_dir / 'confusion_matrix_2017_final.png'}")
    plt.close()
else:
    print("\nNo 2017 test data found in the final test set.")

# Evaluate 2018 data
if len(indices_2018) > 0:
    print(f"\nEvaluating on 2018 test set ({len(indices_2018)} samples)...")
    x_test_ocsvm_2018 = test['ae_s'][indices_2018] # OCSVM-scaled
    x_test_rf_2018 = test['stage2_s'][indices_2018] # RF-scaled
    y_true_2018 = test['y_unknown'][indices_2018]

    y_pred_2018, metrics_2018 = evaluate_pipeline(
        x_test_ocsvm_2018, x_test_rf_2018, y_true_2018, ocsvm_model_100k, rf_model_extra_feature, thresholds
    )
    print("2018 test set metrics:")
    for metric_name, metric_value in metrics_2018.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # Plot confusion matrix for 2018
    classes_2018 = sorted(list(np.unique(np.concatenate((y_true_2018, y_pred_2018)))))
    plot_labels_2018 = sorted(list(set(base_classes + classes_2018)))
    plt.figure(figsize=(10, 8))
    util.plot_confusion_matrix(
        y_true_2018,
        y_pred_2018,
        values=plot_labels_2018,
        labels=plot_labels_2018,
        title="Final Pipeline Performance (2018 Test Data)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_2018_final.png")
    print(f"Saved 2018 confusion matrix to {output_dir / 'confusion_matrix_2018_final.png'}")
    plt.close()
else:
    print("\nNo 2018 test data found in the final test set.")

print("\nFinal Evaluation finished.")

# ============================ 9. SAVE FINAL METADATA ============================
# This section remains the same - saves info about the trained models and final metrics

print("\n=============== Saving Final Metadata ===============")

# Save selected feature names (if available)
if 'selected_features' in locals():
    with open(output_dir / "selected_features.pkl", "wb") as f:
        pickle.dump(selected_features, f)
    print(f"Saved selected features to {output_dir / 'selected_features.pkl'}")
else:
     print("Selected feature names not available to save.")


# Save model architecture info and metrics
model_info = {
    'stage1': {
        'type': 'OneClassSVM',
        'pipeline_steps': ['Imputer', 'PCA', 'OneClassSVM'],
        'parameters': {
            'pca_components': ocsvm_model_100k.named_steps['pca'].n_components if 'pca' in ocsvm_model_100k.named_steps else 'N/A',
            'kernel': ocsvm_model_100k.named_steps['ocsvm'].kernel if 'ocsvm' in ocsvm_model_100k.named_steps else 'N/A',
            'gamma': ocsvm_model_100k.named_steps['ocsvm'].gamma if 'ocsvm' in ocsvm_model_100k.named_steps else 'N/A',
            'nu': ocsvm_model_100k.named_steps['ocsvm'].nu if 'ocsvm' in ocsvm_model_100k.named_steps else 'N/A'
        }
    },
    'stage2': {
        'type': 'RandomForestClassifier',
        'parameters': {
            'n_estimators': rf_model_extra_feature.n_estimators,
            'max_features': rf_model_extra_feature.max_features, # Save actual value used
            'classes': list(rf_model_extra_feature.classes_)
        },
        'extra_feature': 'OCSVM Anomaly Score'
    },
    'thresholds': thresholds,
    'metrics': {
        'full': metrics_full,
        '2017': metrics_2017, # Will be None if no 2017 data
        '2018': metrics_2018  # Will be None if no 2018 data
    },
    'data_sampling': {
        'benign_source': 'Full Combined 2017+2018',
        'malicious_source': 'Combined 2017+2018',
        'malicious_sampling_per_original_type': MALICIOUS_SAMPLE_SIZE_PER_TYPE,
        'split_proportions': {'train': train_prop, 'val': val_prop, 'test': test_prop},
        'stage1_train_benign_samples': {'ocsvm': n_ocsvm_train, 'ae': n_ae_train},
        'stage2_train_malicious_source': 'Train split of sampled malicious data',
        'stage2_val_benign_samples': n_benign_val_stage2
    }
}

with open(output_dir / "model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)
print(f"Saved model info to {output_dir / 'model_info.pkl'}")

print("\n=============== Training Complete ===============")
print(f"All models and metadata saved to {output_dir}/")
print("To use this model, load stage1_ocsvm.p and stage2_rf.p along with the scalers and thresholds.")