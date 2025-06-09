import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score

# --- Reproducibility ---
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
# Note: TensorFlow seed setting is not needed here as we are using scikit-learn

# --- Configuration ---
DATA1_DIR = 'W9/data1/clean/' # CIC-IDS-2017 (Training)
DATA2_DIR = 'W9/data2/clean/' # CSE-CIC-IDS-2018 (Testing)
DROP_COLS = ['Timestamp', 'Label', 'Destination Port'] # Columns to drop

# --- Risk Level Mapping ---
# Define mapping from original labels to risk levels. This is subjective.
# Add ALL labels from BOTH datasets here.
RISK_MAP = {
    # Dataset 1 Labels (CIC-IDS-2017)
    'Benign': 'LOW',
    'DoS Hulk': 'HIGH',
    'DDoS': 'HIGH',
    'PortScan': 'MEDIUM',
    'DoS GoldenEye': 'HIGH',
    'FTP-Patator': 'MEDIUM',
    'DoS slowloris': 'HIGH',
    'DoS Slowhttptest': 'HIGH',
    'SSH-Patator': 'MEDIUM',
    'Bot': 'HIGH',
    'Web Attack  Brute Force': 'HIGH',
    'Web Attack ï¿½ Brute Force': 'HIGH', # Handle encoding issue if present
    'Web Attack  XSS': 'HIGH',
    'Web Attack ï¿½ XSS': 'HIGH', # Handle encoding issue if present
    'Infiltration': 'CRITICAL',
    'Heartbleed': 'CRITICAL',
    'Web Attack  Sql Injection': 'HIGH',
    'Web Attack ï¿½ Sql Injection': 'HIGH', # Handle encoding issue if present

    # Dataset 2 Labels (CSE-CIC-IDS-2018)
    # 'Benign' is already mapped
    'DDoS attacks-LOIC-HTTP': 'HIGH',
    'DDOS attack-HOIC': 'HIGH',
    'DoS attacks-Hulk': 'HIGH', # Note: slightly different name from dataset 1
    # 'Bot' is already mapped
    'Infilteration': 'CRITICAL', # Note: Typo in original label?
    'SSH-Bruteforce': 'MEDIUM', # Note: slightly different name from dataset 1
    'DoS attacks-GoldenEye': 'HIGH', # Note: slightly different name from dataset 1
    'DoS attacks-Slowloris': 'HIGH', # Note: slightly different name from dataset 1
    'DDOS attack-LOIC-UDP': 'HIGH',
    'Brute Force -Web': 'MEDIUM', # Note: slightly different name from dataset 1
    'Brute Force -XSS': 'MEDIUM', # Note: slightly different name from dataset 1
    'SQL Injection': 'HIGH', # Note: slightly different name from dataset 1
    'FTP-BruteForce': 'MEDIUM', # Note: slightly different name from dataset 1
    'DoS attacks-SlowHTTPTest': 'HIGH' # Note: slightly different name from dataset 1
}

# Define the order for reports and confusion matrices
RISK_LEVELS = ['UNKNOWN', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

# --- Data Loading Function ---
def load_and_preprocess_data(data_dir, risk_map, drop_cols):
    """Loads data, applies risk mapping, and separates features/target."""
    file_path = os.path.join(data_dir, 'all_data.parquet')
    print(f"Reading data from: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    print(f"Original shape: {df.shape}")

    # Handle potential encoding issues in labels before mapping
    # Example: df['Label'] = df['Label'].str.replace('ï¿½', '-')

    # Apply Risk Mapping
    df['Risk'] = df['Label'].map(risk_map).fillna('UNKNOWN') # Map labels, fill unmapped as UNKNOWN

    print("Risk level distribution:")
    print(df['Risk'].value_counts())

    # Separate Features (X) and Target (y)
    try:
        X = df.drop(columns=drop_cols + ['Risk'], errors='ignore') # Drop specified cols and new Risk col
        y = df['Risk']
    except KeyError as e:
        print(f"Error dropping columns: {e}. Available columns: {df.columns.tolist()}")
        return None, None

    # Ensure all feature columns are numeric (handle potential errors)
    X = X.select_dtypes(include=np.number)
    print(f"Features shape after selecting numeric: {X.shape}")

    return X, y

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Data
    X_train_raw, y_train = load_and_preprocess_data(DATA1_DIR, RISK_MAP, DROP_COLS)
    X_test_raw, y_test = load_and_preprocess_data(DATA2_DIR, RISK_MAP, DROP_COLS)

    if X_train_raw is None or X_test_raw is None:
        print("Failed to load data. Exiting.")
        exit()

    # Ensure columns match (important for cross-dataset) - use intersection
    common_features = X_train_raw.columns.intersection(X_test_raw.columns)
    print(f"Using {len(common_features)} common features.")
    X_train_raw = X_train_raw[common_features]
    X_test_raw = X_test_raw[common_features]

    # 2. Preprocessing Pipeline
    # Using StandardScaler for potentially better cross-dataset generalization than QuantileTransformer
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # Handle NaN values
        ('scaler', StandardScaler())                 # Scale features
    ])

    # 3. Fit Preprocessor on Training Data ONLY
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train_raw)

    # 4. Transform Both Datasets
    print("Transforming training data...")
    X_train = preprocessor.transform(X_train_raw)
    print("Transforming testing data...")
    X_test = preprocessor.transform(X_test_raw)

    # 5. Define and Train Model
    # Using RandomForest with balanced class weights
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,       # Default, can be tuned
        random_state=seed_value,
        class_weight='balanced', # Important for imbalanced risk levels
        n_jobs=-1               # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 6. Predict on Test Data
    print("\nPredicting on test data...")
    y_pred = model.predict(X_test)

    # 7. Evaluate Model
    print("\n--- Test Set Evaluation ---")

    # Classification Report
    print("Classification Report:")
    # Use zero_division=0 to avoid warnings for labels with no support in predictions
    print(classification_report(y_test, y_pred, labels=RISK_LEVELS, zero_division=0))

    # Balanced Accuracy
    b_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy Score: {b_acc:.4f}")

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=RISK_LEVELS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RISK_LEVELS)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title('Confusion Matrix - Test Set (Dataset 2)')
        plt.tight_layout()
        # Save or show the plot
        plt.savefig('confusion_matrix_risk_cross_dataset.png')
        print("Confusion matrix saved to 'confusion_matrix_risk_cross_dataset.png'")
        # plt.show() # Uncomment to display the plot interactively
    except Exception as e:
        print(f"Could not generate confusion matrix plot: {e}")

    print("\nScript finished.")
