import pandas as pd
import numpy as np
import pickle
import pathlib
import sys
import pyarrow.parquet as pq  # For efficient reading of just the schema

# --- Configuration ---
MODEL_DIR = pathlib.Path("W10/models1")  # Directory with selected indices
DATA1_PATH = pathlib.Path("W9/data1/clean/all_data.parquet")  # Training data
DATA2_PATH = pathlib.Path("W9/data2/clean/all_data.parquet")  # Testing data
LABEL_COLUMN = "Label"  # Name of the label column
INDICES_FILE = MODEL_DIR / "selected_feature_indices.pkl"

print(f"Checking feature names between:\n - Training: {DATA1_PATH}\n - Testing: {DATA2_PATH}")
print(f"Using selected indices from: {INDICES_FILE}")

def get_columns(file_path):
    """Read only column names from a parquet file using pyarrow"""
    try:
        schema = pq.read_schema(file_path)
        return schema.names
    except Exception as e:
        print(f"Error reading schema from {file_path}: {e}")
        return None

# --- Load Selected Indices ---
try:
    with open(INDICES_FILE, "rb") as f:
        selected_indices = pickle.load(f)
    print(f"\nLoaded {len(selected_indices)} selected feature indices: {selected_indices}")
except Exception as e:
    print(f"Error loading indices: {e}")
    sys.exit(1)

# --- Get Column Names from Data1 (Training) ---
data1_columns = get_columns(DATA1_PATH)
if not data1_columns:
    print(f"Could not read columns from {DATA1_PATH}")
    sys.exit(1)

# --- Get Column Names from Data2 (Testing) ---
data2_columns = get_columns(DATA2_PATH)
if not data2_columns:
    print(f"Could not read columns from {DATA2_PATH}")
    sys.exit(1)

# --- Get Feature Names (excluding label column) ---
data1_features = [col for col in data1_columns if col != LABEL_COLUMN]
data2_features = [col for col in data2_columns if col != LABEL_COLUMN]

print(f"\nTotal features in training data: {len(data1_features)}")
print(f"Total features in testing data: {len(data2_features)}")

# --- Get Selected Feature Names from Data1 ---
try:
    selected_feature_names = [data1_features[i] for i in selected_indices]
    print(f"\nSelected features by name from training data:")
    for i, name in enumerate(selected_feature_names):
        print(f"{i+1}. {name} (index {selected_indices[i]})")
except IndexError:
    print(f"\nERROR: Some indices are out of bounds for training data features!")
    print(f"Max valid index is {len(data1_features)-1}, but found index {max(selected_indices)}")
    sys.exit(1)

# --- Check if These Features Exist in Data2 ---
missing_features = [name for name in selected_feature_names if name not in data2_features]

if not missing_features:
    print(f"\n✅ SUCCESS: All {len(selected_feature_names)} selected features exist in both datasets!")
else:
    print(f"\n❌ ERROR: {len(missing_features)} selected features are missing in the testing data:")
    for name in missing_features:
        print(f"  - Missing: {name}")
    
    # If features are missing, also check if any names are similar (potential typos)
    print("\nPotential matches in testing data (for manual verification):")
    for missing in missing_features:
        potential_matches = [name for name in data2_features if any(
            part in name.lower() for part in missing.lower().split())][:5]
        if potential_matches:
            print(f"  For '{missing}', found: {', '.join(potential_matches)}")
        else:
            print(f"  For '{missing}', no similar names found")

# --- Output Feature Order Comparison ---
print("\n--- Feature List Comparison ---")
print(f"Data1 first 5 features: {data1_features[:5]}")
print(f"Data2 first 5 features: {data2_features[:5]}")

# --- Check if using indices would select the same features in both datasets ---
if len(data2_features) <= max(selected_indices):
    print(f"\nWARNING: Testing data has fewer features ({len(data2_features)}) than needed by indices (max index: {max(selected_indices)})")
else:
    data2_selected_by_indices = [data2_features[i] for i in selected_indices]
    print("\nFeatures that would be selected by indices in testing data:")
    for i, name in enumerate(data2_selected_by_indices):
        same = "✓" if name == selected_feature_names[i] else "✗"
        print(f"{i+1}. {name} {same} (expected: {selected_feature_names[i]})")