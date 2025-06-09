# Seed value
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import timeit # Import timeit module
import warnings
from tensorflow import keras # Import keras

# --- FIX: Load the feature list saved during training ---
# It's highly recommended to load the exact feature list used during training
# to ensure consistency. If you saved it during train1.py, load it here.
# Example:
expected_features = None # Initialize
try:
    with open("W9/models1/feature_list.pkl", "rb") as f:
        expected_features = pickle.load(f)
    print(f"Loaded {len(expected_features)} expected features.")
except FileNotFoundError:
    print("Warning: feature_list.pkl not found. Assuming feature order based on input file (risk of inconsistency).")
    expected_features = None # Or define a default list based on your knowledge if the file is missing
except Exception as e:
    print(f"Error loading feature_list.pkl: {e}")
    expected_features = None

# --- FIX: Define comprehensive label mapping ---
# Map all unique labels from BOTH datasets (2017 and 2018) to the target categories
# Target categories: ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
label_mapping = {
    # Benign
    'Benign': 'Benign',

    # DDoS / DoS (from both datasets)
    'DDoS attacks-LOIC-HTTP': '(D)DOS', # 2018
    'DDOS attack-HOIC': '(D)DOS',       # 2018
    'DoS attacks-Hulk': '(D)DOS',       # 2018
    'DoS attacks-GoldenEye': '(D)DOS',  # 2018
    'DoS attacks-Slowloris': '(D)DOS',  # 2018
    'DoS attacks-SlowHTTPTest': '(D)DOS',# 2018
    'DDOS attack-LOIC-UDP': '(D)DOS',   # 2018
    'DoS Hulk': '(D)DOS',               # 2017
    'DDoS': '(D)DOS',                   # 2017
    'DoS slowloris': '(D)DOS',          # 2017
    'DoS Slowhttptest': '(D)DOS',       # 2017
    'DoS GoldenEye': '(D)DOS',          # 2017

    # Botnet (from both datasets)
    'Bot': 'Botnet',

    # Brute Force (from both datasets)
    'FTP-BruteForce': 'Brute Force',    # 2018
    'SSH-Bruteforce': 'Brute Force',    # 2018
    'Brute Force -Web': 'Brute Force',  # 2018 (Assuming this is Brute Force)
    'SSH-Patator': 'Brute Force',       # 2017
    'FTP-Patator': 'Brute Force',       # 2017

    # Port Scan (from 2017)
    'PortScan': 'Port Scan',

    # Web Attack (from both datasets, handling encoding issues)
    'Brute Force -XSS': 'Web Attack',   # 2018 (XSS is Web Attack)
    'SQL Injection': 'Web Attack',      # 2018
    'Web Attack \x96 Brute Force': 'Web Attack', # 2017
    'Web Attack \x96 Sql Injection': 'Web Attack', # 2017
    'Web Attack \x96 XSS': 'Web Attack', # 2017
    'Web Attack ï¿½ Brute Force': 'Web Attack', # 2017 (Treat as same)
    'Web Attack ï¿½ Sql Injection': 'Web Attack', # 2017 (Treat as same)
    'Web Attack ï¿½ XSS': 'Web Attack', # 2017 (Treat as same)

    # Infiltration / Unknown / Zero-Day (Map to Unknown)
    'Infilteration': 'Unknown',         # 2018
    'Infiltration': 'Unknown',          # 2017

    # Heartbleed (Map to Unknown as rare/specific)
    'Heartbleed': 'Unknown',            # 2017
}

SAMPLE_SIZE = 20000 # Define the desired sample size

#  Load Test Data (Using CIC-IDS-2017 for main evaluation)
print("\nLoading CIC-IDS-2017 data...")
full_test_df = pd.read_parquet("W9/data1/clean/all_data.parquet")
print(f"Original CIC-IDS-2017 shape: {full_test_df.shape}")

# --- Stratified Sampling for CIC-IDS-2017 ---
# Use train_test_split to get a stratified sample. We only keep the 'train' part.
# Ensure the label column exists before stratifying
if 'Label' in full_test_df.columns:
    test_df, _ = train_test_split(
        full_test_df,
        train_size=min(SAMPLE_SIZE, len(full_test_df)), # Ensure sample size isn't larger than dataset
        stratify=full_test_df['Label'],
        random_state=seed_value
    )
    print(f"Sampled CIC-IDS-2017 shape: {test_df.shape}")
else:
    print("Error: 'Label' column not found in CIC-IDS-2017 data for stratification.")
    # Fallback to random sampling or exit
    test_df = full_test_df.sample(n=min(SAMPLE_SIZE, len(full_test_df)), random_state=seed_value)
    print(f"Warning: Using random sampling for CIC-IDS-2017. Sampled shape: {test_df.shape}")
# --- End Sampling ---

print("Original Test Labels (Sampled CIC-IDS-2017):")
print(test_df["Label"].value_counts())

# --- FIX: Identify and drop non-numeric columns ---
# Based on your read_file output, 'Timestamp' is datetime
non_numeric_cols = ['Label', 'Timestamp']
# Check if columns exist before dropping
cols_to_drop = [col for col in non_numeric_cols if col in test_df.columns]
x = test_df.drop(columns=cols_to_drop)
print(f"\nDropped columns: {cols_to_drop}. Input 'x' shape before consistency check: {x.shape}")

# --- FIX: Ensure feature consistency ---
PIPELINE_EXPECTED_FEATURES = 67 # Define the number of features the pipeline expects

if expected_features:
    # If feature list is loaded, use it
    if len(expected_features) != PIPELINE_EXPECTED_FEATURES:
        print(f"Error: Loaded feature list has {len(expected_features)} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
        exit() # Or handle error appropriately

    missing_features = [feat for feat in expected_features if feat not in x.columns]
    extra_features = [col for col in x.columns if col not in expected_features]

    if missing_features:
        print(f"Error: Test data 'x' is missing expected features: {missing_features}")
        exit() # Or handle error appropriately

    if extra_features or list(x.columns) != expected_features:
        print(f"Adjusting 'x' columns to match expected {len(expected_features)} features (Order and count).")
        try:
            x = x[expected_features] # Select and reorder features
        except KeyError as e:
            print(f"Error selecting features based on expected_features list: {e}")
            print("Check if expected_features list matches columns available after dropping non-numeric ones.")
            exit()
    else:
        print("Test data 'x' features match expected features.")

elif x.shape[1] == PIPELINE_EXPECTED_FEATURES:
    # If feature list is NOT loaded, but x already has the correct number of features
    print(f"Warning: feature_list.pkl not found. 'x' already has {PIPELINE_EXPECTED_FEATURES} features. Assuming they are the correct ones and in the correct order (RISKY).")
    # No changes needed to x in this specific case

elif x.shape[1] > PIPELINE_EXPECTED_FEATURES:
     # If feature list is NOT loaded, and x has MORE features than expected
     print(f"Warning: feature_list.pkl not found. 'x' has {x.shape[1]} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
     # --- MODIFIED FALLBACK (Based on ValueError) ---
     # The error indicates 'Destination Port' should be dropped, not the last column.
     # This is still RISKY as it assumes the rest of the order matches training.
     if 'Destination Port' in x.columns and x.shape[1] == 68 and PIPELINE_EXPECTED_FEATURES == 67:
         print("Attempting to drop 'Destination Port' as a fallback (RISKY).")
         try:
             x = x.drop(columns=['Destination Port'])
             print(f"Shape of 'x' adjusted to: {x.shape}")
         except KeyError:
              print("Fallback failed: 'Destination Port' not found to drop.")
              exit()
     else:
          # Original risky fallback if the specific condition isn't met
          print(f"Attempting to drop the last {x.shape[1] - PIPELINE_EXPECTED_FEATURES} feature(s) as a fallback (VERY RISKY).")
          x = x.iloc[:, :PIPELINE_EXPECTED_FEATURES]
          print(f"Shape of 'x' adjusted to: {x.shape}")
     # --- END MODIFIED FALLBACK ---

else: # x.shape[1] < PIPELINE_EXPECTED_FEATURES
    # If feature list is NOT loaded, and x has FEWER features than expected
    print(f"Error: feature_list.pkl not found and 'x' has only {x.shape[1]} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
    exit() # Cannot proceed

# --- End FIX ---

print(f"Final shape of 'x' passed to models: {x.shape}") # Verify shape before use

# Verify dtypes in x (optional)
# print("\nData types for model input 'x':")
# print(x.dtypes.value_counts())

# Apply the comprehensive mapping to CIC-IDS-2017 labels
# ... (rest of the script) ...

# Verify dtypes in x (optional)
# print("\nData types for model input 'x':")
# print(x.dtypes.value_counts())

# Apply the comprehensive mapping to CIC-IDS-2017 labels
y_true_raw = test_df["Label"]
y_true_mapped = y_true_raw.map(label_mapping).fillna('Unknown') # Map labels, default unmapped to Unknown

print("\nMapped Test Labels (CIC-IDS-2017):")
print(y_true_mapped.value_counts())


#  Load additional data from CSE-CIC-IDS-2018 (Optional Robustness Check)
print("\nLoading CSE-CIC-IDS-2018 data for robustness check...")
full_df_2018 = pd.read_parquet("W9/data2/clean/all_data.parquet") # Use the correct file
print(f"Original CSE-CIC-IDS-2018 shape: {full_df_2018.shape}")

# --- Stratified Sampling for CSE-CIC-IDS-2018 ---
if 'Label' in full_df_2018.columns:
    df_2018, _ = train_test_split(
        full_df_2018,
        train_size=min(SAMPLE_SIZE, len(full_df_2018)),
        stratify=full_df_2018['Label'],
        random_state=seed_value
    )
    print(f"Sampled CSE-CIC-IDS-2018 shape: {df_2018.shape}")
else:
    print("Error: 'Label' column not found in CSE-CIC-IDS-2018 data for stratification.")
    df_2018 = full_df_2018.sample(n=min(SAMPLE_SIZE, len(full_df_2018)), random_state=seed_value)
    print(f"Warning: Using random sampling for CSE-CIC-IDS-2018. Sampled shape: {df_2018.shape}")
# --- End Sampling ---

y_18_raw = df_2018['Label'] # Keep original labels for this check

# ... (previous code) ...

# --- FIX: Also drop non-numeric columns from x_18 ---
# Check if columns exist before dropping
cols_to_drop_18 = [col for col in non_numeric_cols if col in df_2018.columns]
x_18 = df_2018.drop(columns=cols_to_drop_18)
print(f"\nDropped columns: {cols_to_drop_18}. Input 'x_18' shape before consistency check: {x_18.shape}") # Modified print

# --- FIX: Ensure feature consistency for x_18 (Apply same logic as for x) ---
if expected_features:
    # If feature list is loaded, use it
    if len(expected_features) != PIPELINE_EXPECTED_FEATURES:
        print(f"Error: Loaded feature list has {len(expected_features)} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
        exit()

    missing_features_18 = [feat for feat in expected_features if feat not in x_18.columns]
    extra_features_18 = [col for col in x_18.columns if col not in expected_features]

    if missing_features_18:
        print(f"Error: 2018 data 'x_18' is missing expected features: {missing_features_18}")
        exit()

    if extra_features_18 or list(x_18.columns) != expected_features:
        print(f"Adjusting 'x_18' columns to match expected {len(expected_features)} features (Order and count).")
        try:
            x_18 = x_18[expected_features] # Select and reorder features
        except KeyError as e:
            print(f"Error selecting features for 'x_18' based on expected_features list: {e}")
            exit()
    else:
        print("2018 data 'x_18' features match expected features.")

elif x_18.shape[1] == PIPELINE_EXPECTED_FEATURES:
    # If feature list is NOT loaded, but x_18 already has the correct number of features
    print(f"Warning: feature_list.pkl not found. 'x_18' already has {PIPELINE_EXPECTED_FEATURES} features. Assuming correct order (RISKY).")

elif x_18.shape[1] > PIPELINE_EXPECTED_FEATURES:
     # If feature list is NOT loaded, and x_18 has MORE features than expected
     print(f"Warning: feature_list.pkl not found. 'x_18' has {x_18.shape[1]} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
     # --- Apply MODIFIED FALLBACK to x_18 ---
     if 'Destination Port' in x_18.columns and x_18.shape[1] == 68 and PIPELINE_EXPECTED_FEATURES == 67:
         print("Attempting to drop 'Destination Port' from 'x_18' as a fallback (RISKY).")
         try:
             x_18 = x_18.drop(columns=['Destination Port'])
             print(f"Shape of 'x_18' adjusted to: {x_18.shape}")
         except KeyError:
              print("Fallback failed for 'x_18': 'Destination Port' not found.")
              exit()
     else:
          print(f"Attempting to drop the last {x_18.shape[1] - PIPELINE_EXPECTED_FEATURES} feature(s) from 'x_18' as a fallback (VERY RISKY).")
          x_18 = x_18.iloc[:, :PIPELINE_EXPECTED_FEATURES]
          print(f"Shape of 'x_18' adjusted to: {x_18.shape}")
     # --- END FALLBACK for x_18 ---

else: # x_18.shape[1] < PIPELINE_EXPECTED_FEATURES
    print(f"Error: feature_list.pkl not found and 'x_18' has only {x_18.shape[1]} features, but pipeline expects {PIPELINE_EXPECTED_FEATURES}.")
    exit()
# --- End FIX ---

print(f"Final shape of 'x_18' passed to models: {x_18.shape}") # Verify shape

print("\nOriginal Labels (CSE-CIC-IDS-2018 Sampled):") # Modified print
print(y_18_raw.value_counts())

# Load Models
print("\nLoading models...")
# --- FIX: Use try-except for loading ---
try:
    # Optimized pipelines
    with open("W9/models1/stage1_ocsvm.p","rb") as f:
        stage1 = pickle.load(f) # OCSVM Pipeline (Scaler+PCA+OCSVM)
    with open("W9/models1/stage2_rf.p","rb") as f:
        stage2 = pickle.load(f) # RF Model (trained with extra feature)

    # Individual feature scalers and classification models
    # with open("W9/models1/stage1_ocsvm_model.p","rb") as f:
    #     stage1_model = pickle.load(f) # Just OCSVM (Optional, not used in main predict)
    with open("W9/models1/stage1_ocsvm_scaler.p","rb") as f:
        stage1_scaler = pickle.load(f) # Scaler for OCSVM input (Used in SotA)
    # with open("W9/models1/stage2_rf_model.p","rb") as f:
    #     stage2_model = pickle.load(f) # Same as stage2 RF model (Optional)
    with open("W9/models1/stage2_rf_scaler.p","rb") as f:
        stage2_scaler = pickle.load(f) # Scaler for original features (Used in SotA)

    # RF baseline model and feature scaler
    with open("W9/models1/baseline_rf.p","rb") as f:
        baseline_rf = pickle.load(f) # Baseline RF model
    with open("W9/models1/baseline_rf_scaler.p","rb") as f:
        baseline_rf_scaler = pickle.load(f) # Scaler for baseline RF input

    # Optimized models for Bovenzi et al.
    try:
        sota_stage1 = keras.models.load_model("W9/models1/sota_stage1.h5") # Load Keras AE
        print("Loaded sota_stage1.h5")
    except OSError as e:
        print(f"Error loading Keras model W9/models1/sota_stage1.h5: {e}")
        print("Ensure the .h5 file exists and is a valid Keras model file.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading sota_stage1.h5: {e}")
        exit()

    try:
        with open("W9/models1/sota_stage2.p","rb") as f: # Load SotA RF model
            sota_stage2 = pickle.load(f)
        print("Loaded sota_stage2.p")
    except FileNotFoundError:
        print("Error: W9/models1/sota_stage2.p not found.")
        exit()
    except Exception as e:
        # Catch potential version incompatibility
        print(f"An error occurred loading sota_stage2.p: {e}")
        print("This might be due to scikit-learn version mismatch. Ensure model was trained/saved with compatible version.")
        exit()

except FileNotFoundError as e:
    print(f"Error loading a required model file: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    exit()
print("Models loaded successfully.")

#  Thresholds (Example: using "balanced" thresholds from original paper/tuning)
# These might need re-tuning based on the specific dataset and mapping used
tau_b_bal = -0.0002196942507948895
tau_m_bal = 0.98
tau_u_bal = 0.0040588613744241275

#  Evaluation of Time Complexity
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore numpy/tf future warnings if any

# hids_predict: Function to perform classification of all stages combined of the novel hierarchical
# multi-stage intrusion detection approach by Verkerken et al.

# hids_sota_predict: Function to evaluate former SotA approach existing of two stages by Bovenzi et al.


# --- FIX: Correct input preparation for stage2 in hids_predict ---
# Ensure models expect 68 features (or adjust based on training)
def hids_predict(x_data, tau_b, tau_m, tau_u):
    # Stage 1: Use the OCSVM pipeline directly
    # The pipeline handles scaling/PCA internally
    print("  hids_predict: Running Stage 1 (OCSVM)...")
    proba_1 = -stage1.decision_function(x_data) # invert sign to act as anomaly score
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    print(f"  hids_predict: Stage 1 counts - {np.unique(pred_1, return_counts=True)}")

    attack_mask = (pred_1 == "Attack")
    if np.any(attack_mask):
        print("  hids_predict: Running Stage 2 (RF)...")
        # Stage 2 Input Preparation:
        # 1. Scale original features using the correct scaler (baseline_rf_scaler)
        x_data_attack = x_data[attack_mask]
        # Ensure scaler expects the same number of features as x_data_attack
        if baseline_rf_scaler.n_features_in_ != x_data_attack.shape[1]:
             print(f"Error: baseline_rf_scaler expects {baseline_rf_scaler.n_features_in_} features, but got {x_data_attack.shape[1]}")
             # Handle error
        x_scaled_orig_features = baseline_rf_scaler.transform(x_data_attack) # Use scaler fitted on original features
        # 2. Get OCSVM scores for the 'Attack' samples (already calculated as proba_1)
        proba_1_attack = proba_1[attack_mask]
        # 3. Combine scaled features and OCSVM score
        stage2_input = np.column_stack((x_scaled_orig_features, proba_1_attack))
        print(f"  hids_predict: Stage 2 input shape - {stage2_input.shape}")

        # Stage 2 Prediction: Use the combined input
        # Ensure stage2 model expects the same number of features as stage2_input
        if stage2.n_features_in_ != stage2_input.shape[1]:
             print(f"Error: stage2 RF model expects {stage2.n_features_in_} features, but got {stage2_input.shape[1]}")
             # Handle error
        proba_2 = stage2.predict_proba(stage2_input)
        pred_2 = np.where(
            np.max(proba_2, axis=1) > tau_m,
            stage2.classes_[np.argmax(proba_2, axis=1)],
            "Unknown")
        print(f"  hids_predict: Stage 2 counts (among attacks) - {np.unique(pred_2, return_counts=True)}")


        # Stage 3 (Extension): Use OCSVM scores for samples predicted as 'Unknown' by Stage 2
        unknown_mask_stage2 = (pred_2 == "Unknown")
        if np.any(unknown_mask_stage2):
            print("  hids_predict: Running Stage 3 (Thresholding)...")
            # Get original anomaly scores for samples predicted as Attack then Unknown
            proba_3 = proba_1_attack[unknown_mask_stage2] # Use scores already calculated
            pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
            print(f"  hids_predict: Stage 3 counts (among unknowns) - {np.unique(pred_3, return_counts=True)}")
            # Update the 'Unknown' predictions from stage 2 based on stage 3
            pred_2[unknown_mask_stage2] = pred_3

        # Update the 'Attack' predictions from stage 1 with stage 2/3 results
        pred_1[attack_mask] = pred_2
    return pred_1

# --- FIX: hids_sota_predict - Ensure correct scaler usage and feature count ---
def hids_sota_predict(x_data, tau_b, tau_m):
    print("  hids_sota_predict: Running Stage 1 (AE)...")
    # Stage 1 (AE): Scale with stage1_scaler (assuming AE trained on this scaling)
    if stage1_scaler.n_features_in_ != x_data.shape[1]:
        print(f"Error: stage1_scaler expects {stage1_scaler.n_features_in_} features, but got {x_data.shape[1]}")
    x_s_ae = stage1_scaler.transform(x_data) # Check if AE expects this scaling
    # Ensure AE expects the same number of features
    # Keras models don't have n_features_in_ directly, check input_shape if needed
    x_pred_ae = sota_stage1.predict(x_s_ae, verbose=0) # verbose=0 to reduce output
    if x_s_ae.shape != x_pred_ae.shape:
        print(f"Error: AE input shape {x_s_ae.shape} != output shape {x_pred_ae.shape}")
    proba_1 = np.sum((x_s_ae - x_pred_ae)**2, axis=1) # Reconstruction error (N,)
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    print(f"  hids_sota_predict: Stage 1 counts - {np.unique(pred_1, return_counts=True)}")


    attack_mask = (pred_1 == "Attack")
    if np.any(attack_mask):
        print("  hids_sota_predict: Running Stage 2 (RF)...")
        # Prepare input for sota_stage2
        # 1. Get original features for predicted attacks
        x_data_attack = x_data[attack_mask]
        # 2. Scale original features using stage2_scaler
        if stage2_scaler.n_features_in_ != x_data_attack.shape[1]:
            print(f"Error: stage2_scaler expects {stage2_scaler.n_features_in_} features, but got {x_data_attack.shape[1]}")
        x_s_rf = stage2_scaler.transform(x_data_attack) # Shape (M, 68) if x_data has 68 features
        # 3. Get reconstruction errors for predicted attacks
        proba_1_attack = proba_1[attack_mask] # Shape (M,)
        # 4. Combine scaled features and reconstruction error
        sota_stage2_input = np.column_stack((x_s_rf, proba_1_attack)) # Shape (M, 69) if x_s_rf has 68 features
        print(f"  hids_sota_predict: Stage 2 input shape - {sota_stage2_input.shape}")

        # Stage 2 (RF): Predict using the combined input
        # Ensure sota_stage2 model expects the same number of features
        if sota_stage2.n_features_in_ != sota_stage2_input.shape[1]:
             print(f"Error: sota_stage2 RF model expects {sota_stage2.n_features_in_} features, but got {sota_stage2_input.shape[1]}")
             # Handle error
        proba_2 = sota_stage2.predict_proba(sota_stage2_input)
        pred_2 = np.where(
            np.max(proba_2, axis=1) > tau_m,
            sota_stage2.classes_[np.argmax(proba_2, axis=1)], # Use sota_stage2 classes
            "Unknown")
        print(f"  hids_sota_predict: Stage 2 counts (among attacks) - {np.unique(pred_2, return_counts=True)}")
        pred_1[attack_mask] = pred_2
    return pred_1

# --- Use timeit module correctly ---
n_runs = 1 # Reduce runs for quicker testing initially
n_loops = 1
print("\nTiming hids_predict (Balanced Thresholds):")
time_bal = timeit.timeit(lambda: hids_predict(x, tau_b_bal, tau_m_bal, tau_u_bal), number=n_runs * n_loops)
print(f"  Balanced Avg Time:    {time_bal / (n_runs * n_loops):.6f} seconds")


# Baseline RF
print("\nTiming Baseline RF:")
threshold_baseline = 0.43 # This might need tuning
def predict_baseline():
    print("  predict_baseline: Scaling...")
    if baseline_rf_scaler.n_features_in_ != x.shape[1]:
        print(f"Error: baseline_rf_scaler expects {baseline_rf_scaler.n_features_in_} features, but got {x.shape[1]}")
    x_s = baseline_rf_scaler.transform(x)
    print("  predict_baseline: Predicting...")
    if baseline_rf.n_features_in_ != x_s.shape[1]:
        print(f"Error: baseline_rf expects {baseline_rf.n_features_in_} features, but got {x_s.shape[1]}")
    y_proba = baseline_rf.predict_proba(x_s)
    y_pred = np.where(np.max(y_proba, axis=1) > threshold_baseline, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
    print(f"  predict_baseline: Counts - {np.unique(y_pred, return_counts=True)}")
    return y_pred
time_baseline = timeit.timeit(predict_baseline, number=n_runs * n_loops)
print(f"  Baseline RF Avg Time: {time_baseline / (n_runs * n_loops):.6f} seconds")


#  Bovenzi et al.
print("\nTiming Bovenzi et al. (SotA):")
# Thresholds experimentally optimized (might need re-tuning)
tau_b_sota = 0.7580776764761945
tau_m_sota = 0.98
time_sota = timeit.timeit(lambda: hids_sota_predict(x, tau_b_sota, tau_m_sota), number=n_runs * n_loops)
print(f"  SotA Avg Time:        {time_sota / (n_runs * n_loops):.6f} seconds")


# Evaluate Multi-Stage Model (Using chosen "balanced" thresholds for evaluation)
print("\nEvaluating Multi-Stage Model (Balanced Thresholds) on CIC-IDS-2017...")
# Re-run prediction for evaluation (avoid using timed result directly)
y_pred = hids_predict(x, tau_b_bal, tau_m_bal, tau_u_bal)
print("\nFinal Prediction Counts (Multi-Stage):")
print(np.unique(y_pred, return_counts=True))


#  Statistics and Visualizations of the Results
# --- FIX: Clean up plot_confusion_matrix ---
def plot_confusion_matrix(y_true, y_pred_plot, figsize=(10,10), cmap="Blues", values=None, labels=None, title="", ax=None):
    # Ensure labels and values cover all unique items in y_true and y_pred_plot
    all_items = sorted(list(set(y_true) | set(y_pred_plot)))
    if labels is None:
        labels = all_items
    if values is None:
        values = all_items # Use all unique items found for matrix calculation

    # Ensure 'values' used for confusion_matrix includes all items present
    cm = confusion_matrix(y_true, y_pred_plot, labels=values)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    # Handle potential division by zero if a class has no true samples
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                 annot[i, j] = '0' # Avoid showing 0.0% for 0 counts
            else:
                 annot[i, j] = '%.1f%%\n%d' % (p, c)

    # Use 'labels' for display purposes on the plot axes
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm_df, cmap=cmap, annot=annot, fmt='', ax=ax, cbar=False, annot_kws={"size": 8}) # Adjust font size if needed
    ax.set_title(title if title else "Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()


# Define the classes expected by the model/evaluation (target categories)
classes_eval = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']

# --- FIX: Use mapped true labels for plotting and report ---
plot_confusion_matrix(y_true_mapped, y_pred, values=classes_eval, labels=classes_eval, title="Multi-Stage Model Performance on CIC-IDS-2017")
plt.show()

print("\nClassification Report (CIC-IDS-2017):")
# Ensure report includes all relevant labels present in y_true_mapped or y_pred
report_labels = sorted(list(set(y_true_mapped) | set(y_pred)))
print(classification_report(y_true_mapped, y_pred, labels=report_labels, digits=4, zero_division=0))

# Robustness - Perform classification on additional samples from CSE-CIC-IDS-2018
print("\nRobustness Check on CSE-CIC-IDS-2018:")
# Use balanced thresholds for HIDS
print("  Running HIDS...")
y_hids_18 = hids_predict(x_18, tau_b_bal, tau_m_bal, tau_u_bal)
print("HIDS Preds (on 2018):", np.unique(y_hids_18, return_counts=True))

# SotA comparison
print("\n  Running SotA...")
y_sota_18 = hids_sota_predict(x_18, tau_b_sota, tau_m_sota)
print("SotA Preds (on 2018):", np.unique(y_sota_18, return_counts=True))

# Baseline RF comparison
print("\n  Running Baseline RF...")
y_pred_18_baseline = predict_baseline() # Re-use the function, now on x_18
# Need to modify predict_baseline or create a new one for x_18
def predict_baseline_18():
    print("  predict_baseline_18: Scaling...")
    if baseline_rf_scaler.n_features_in_ != x_18.shape[1]:
        print(f"Error: baseline_rf_scaler expects {baseline_rf_scaler.n_features_in_} features, but got {x_18.shape[1]}")
    x_18_s = baseline_rf_scaler.transform(x_18)
    print("  predict_baseline_18: Predicting...")
    if baseline_rf.n_features_in_ != x_18_s.shape[1]:
        print(f"Error: baseline_rf expects {baseline_rf.n_features_in_} features, but got {x_18_s.shape[1]}")
    y_proba_18 = baseline_rf.predict_proba(x_18_s)
    y_pred_18 = np.where(np.max(y_proba_18, axis=1) > threshold_baseline, baseline_rf.classes_[np.argmax(y_proba_18, axis=1)], 'Unknown')
    print(f"  predict_baseline_18: Counts - {np.unique(y_pred_18, return_counts=True)}")
    return y_pred_18

y_pred_18_baseline = predict_baseline_18()
print("Baseline RF Preds (on 2018):", np.unique(y_pred_18_baseline, return_counts=True))


print("\nScript finished.")