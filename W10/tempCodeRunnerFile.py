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