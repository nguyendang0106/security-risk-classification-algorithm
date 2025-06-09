import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F # For Focal Loss definition if needed, though not used in eval
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
from tqdm import tqdm

# ===================== Model and Dataset Definitions (Must match train12.py) =====================

class UNSWDataset(Dataset):
    def __init__(self, x, y_attack, y_risk):
        # Ensure data is tensor
        self.x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        self.y_attack = torch.tensor(y_attack, dtype=torch.long) if not isinstance(y_attack, torch.Tensor) else y_attack
        self.y_risk = torch.tensor(y_risk, dtype=torch.long) if not isinstance(y_risk, torch.Tensor) else y_risk

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_attack[idx], self.y_risk[idx]

class EnhancedMultiTaskNN(nn.Module):
    # Copied directly from train12.py
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes_attack, num_classes_risk, dropout_rate=0.3):
        super(EnhancedMultiTaskNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_size) # Batchnorm on input
        self.shared_l1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.shared_l2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu = nn.LeakyReLU() # Changed to LeakyReLU
        self.dropout = nn.Dropout(dropout_rate)

        # Output heads
        self.attack_out = nn.Linear(hidden_size2, num_classes_attack)
        self.risk_out = nn.Linear(hidden_size2, num_classes_risk)

    def forward(self, x):
        # Handle potential BN issues with batch size 1 during evaluation if batch size is not dropped
        if x.shape[0] == 1 and (isinstance(self.bn_input, nn.BatchNorm1d) or isinstance(self.bn1, nn.BatchNorm1d) or isinstance(self.bn2, nn.BatchNorm1d)):
             # Temporarily switch to eval mode for single samples if needed, though usually handled by model.eval()
             pass # model.eval() should handle this correctly

        x = self.bn_input(x) # Apply BN to input
        x = self.relu(self.bn1(self.shared_l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.shared_l2(x)))
        x = self.dropout(x)
        out_attack = self.attack_out(x)
        out_risk = self.risk_out(x)
        return out_attack, out_risk

# ===================== Evaluation Utilities =====================

def plot_confusion_matrix_eval(y_true, y_pred, class_names, title, filename):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

def evaluate_model(model, test_loader, device, attack_class_names, risk_class_names, model_desc="Enhanced Model"):
    """Evaluates the multi-task model on the test set."""
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout, uses running stats for BN)

    all_y_attack_true = []
    all_y_risk_true = []
    all_y_attack_pred = []
    all_y_risk_pred = []

    print(f"\n--- Evaluating Model: {model_desc} on Test Set ---")
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x, y_attack, y_risk in progress_bar:
            x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)

            out_attack, out_risk = model(x)
            _, pred_attack = torch.max(out_attack, 1)
            _, pred_risk = torch.max(out_risk, 1)

            all_y_attack_true.extend(y_attack.cpu().numpy())
            all_y_risk_true.extend(y_risk.cpu().numpy())
            all_y_attack_pred.extend(pred_attack.cpu().numpy())
            all_y_risk_pred.extend(pred_risk.cpu().numpy())

    # --- Attack Classification Metrics ---
    print("\n--- Attack Classification Results ---")
    attack_acc = accuracy_score(all_y_attack_true, all_y_attack_pred)
    attack_f1_weighted = f1_score(all_y_attack_true, all_y_attack_pred, average='weighted', zero_division=0)
    attack_f1_macro = f1_score(all_y_attack_true, all_y_attack_pred, average='macro', zero_division=0)
    print(f"Accuracy: {attack_acc:.4f}")
    print(f"F1 Score (Weighted): {attack_f1_weighted:.4f}")
    print(f"F1 Score (Macro): {attack_f1_macro:.4f}")
    print("Classification Report:")
    attack_labels = np.arange(len(attack_class_names))
    print(classification_report(all_y_attack_true, all_y_attack_pred, labels=attack_labels, target_names=attack_class_names, zero_division=0))
    plot_confusion_matrix_eval(all_y_attack_true, all_y_attack_pred, attack_class_names,
                               f'Attack Confusion Matrix ({model_desc})',
                               f'W6/results/enhanced_attack_cm.png')

    # --- Risk/Severity Classification Metrics ---
    print("\n--- Risk/Severity Classification Results ---")
    risk_acc = accuracy_score(all_y_risk_true, all_y_risk_pred)
    risk_f1_weighted = f1_score(all_y_risk_true, all_y_risk_pred, average='weighted', zero_division=0)
    risk_f1_macro = f1_score(all_y_risk_true, all_y_risk_pred, average='macro', zero_division=0)
    print(f"Accuracy: {risk_acc:.4f}")
    print(f"F1 Score (Weighted): {risk_f1_weighted:.4f}")
    print(f"F1 Score (Macro): {risk_f1_macro:.4f}")
    print("Classification Report:")
    # Create generic risk class names based on max value + 1
    max_risk_val = max(all_y_risk_true + all_y_risk_pred)
    risk_labels = np.arange(max_risk_val + 1)
    risk_class_names_eval = [f"Risk_{i}" for i in risk_labels] # Generate names for report
    print(classification_report(all_y_risk_true, all_y_risk_pred, labels=risk_labels, target_names=risk_class_names_eval, zero_division=0))
    plot_confusion_matrix_eval(all_y_risk_true, all_y_risk_pred, risk_class_names_eval,
                               f'Risk/Severity Confusion Matrix ({model_desc})',
                               f'W6/results/enhanced_risk_cm.png')

    print("-" * 50)


# ===================== Main Evaluation Script =====================
def main():
    print("Starting evaluation script for enhanced model...")
    start_time_eval = time.time()

    # Ensure results directory exists
    os.makedirs('W6/results', exist_ok=True)

    # --- 1. Load and Preprocess Data (Replicate train12.py steps EXACTLY) ---
    print("Loading and preprocessing data...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    try:
        # --- Basic Preprocessing (Label Encoding, Severity Mapping) ---
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)

        severity_map = {
            "Normal": 0, "Analysis": 2, "Backdoor": 3, "DoS": 3, "Exploits": 4,
            "Fuzzers": 1, "Generic": 2, "Reconnaissance": 1, "Shellcode": 4, "Worms": 4
        }
        combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)

        le_attack = LabelEncoder()
        combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
        attack_class_names = list(le_attack.classes_) # Store class names for reporting
        num_attack_classes = len(attack_class_names)
        num_risk_classes = combined_data['severity_encoded'].nunique() # Or max + 1

        combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
        combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
        combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

        # --- Feature Analysis for PCA (Numeric Only - on combined data) ---
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat_encoded')['attack_cat_encoded'].nsmallest(7).index)
        exclude_pca_cols = list(set(lowCORR + lowSTD))
        for col in ['attack_cat', 'attack_cat_encoded', 'severity_encoded']:
             if col in exclude_pca_cols:
                 exclude_pca_cols.remove(col)
        print('Columns identified for PCA reduction:', exclude_pca_cols)

        # --- Prepare Data (Drop original attack_cat) ---
        data_x_full = combined_data.drop(['attack_cat', 'attack_cat_encoded', 'severity_encoded'], axis=1)
        data_y_attack = combined_data['attack_cat_encoded']
        data_y_severity = combined_data['severity_encoded']

        # --- Train/Validation/Test Split (Replicate train12.py split) ---
        print("Replicating Train/Validation/Test split (60/20/20)...")
        # Stratify based on attack category as in training
        X_train_val, X_test, y_train_val_attack, y_test_attack, y_train_val_severity, y_test_severity = train_test_split(
            data_x_full, data_y_attack, data_y_severity, test_size=0.20, random_state=42, stratify=data_y_attack
        )
        # We only need the training set to fit PCA/Scaler, and the test set for evaluation
        # We don't strictly need the validation set here, but we need the train set from the first split
        X_train, _, y_train_attack, _, y_train_severity, _ = train_test_split(
            X_train_val, y_train_val_attack, y_train_val_severity, test_size=0.25, random_state=42, stratify=y_train_val_attack # 0.25 * 0.8 = 0.2
        )
        print(f"Train shape (for fitting): {X_train.shape}, Test shape (for eval): {X_test.shape}")

        # --- Apply PCA (Fit on Train, Transform Test) ---
        print("Applying PCA (fit on train, transform test)...")
        pca = PCA(n_components=3)
        valid_exclude_train = [col for col in exclude_pca_cols if col in X_train.columns]

        # Prepare PCA input (handle NaNs using train mean)
        X_train_pca_input = X_train[valid_exclude_train].copy()
        X_test_pca_input = X_test[valid_exclude_train].copy()
        train_means = X_train_pca_input.mean()
        X_train_pca_input.fillna(train_means, inplace=True)
        X_test_pca_input.fillna(train_means, inplace=True)

        # Fit PCA ONLY on training data
        pca.fit(X_train_pca_input)
        print("PCA Explained variance ratio (from training fit):", sum(pca.explained_variance_ratio_))

        # Transform test set
        test_pca_res = pca.transform(X_test_pca_input)

        # Create PCA dataframe for test set
        pca_cols = ['PCA1', 'PCA2', 'PCA3']
        test_pca_df = pd.DataFrame(test_pca_res, columns=pca_cols, index=X_test.index)

        # --- Prepare final feature set for TEST data ---
        X_test_dropped = X_test.drop(columns=valid_exclude_train)
        X_test_final = pd.concat([X_test_dropped, test_pca_df], axis=1)
        print(f"Test feature shape after PCA combination: {X_test_final.shape}")

        # --- Scaling (Fit on Train, Transform Test) ---
        print("Applying StandardScaler (fit on train, transform test)...")
        # Need to prepare the final *training* features just for fitting the scaler
        train_pca_res = pca.transform(X_train_pca_input) # Transform train PCA input
        train_pca_df = pd.DataFrame(train_pca_res, columns=pca_cols, index=X_train.index)
        X_train_dropped = X_train.drop(columns=valid_exclude_train)
        X_train_final_for_scaler = pd.concat([X_train_dropped, train_pca_df], axis=1)

        scaler = StandardScaler()
        scaler.fit(X_train_final_for_scaler) # Fit scaler ONLY on training data
        X_test_scaled = scaler.transform(X_test_final) # Transform test data

        # Convert test target Series to numpy arrays
        y_test_attack_np = y_test_attack.values
        y_test_severity_np = y_test_severity.values

        # --- Verify final shapes for test set ---
        print("\n--- Verifying Shapes Before Test Dataset Creation ---")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_test_attack_np shape: {y_test_attack_np.shape}")
        print(f"y_test_severity_np shape: {y_test_severity_np.shape}")
        assert X_test_scaled.shape[0] == y_test_attack_np.shape[0] == y_test_severity_np.shape[0], "Test data shapes mismatch!"
        print("--- Test Shape Verification Passed ---")

        # Clean up intermediate dataframes
        del train, test, combined_data, data_x_full, X_train_val, y_train_val_attack, y_train_val_severity
        del X_train, y_train_attack, y_train_severity, X_test
        del X_train_pca_input, X_test_pca_input, train_pca_df, test_pca_df, X_train_dropped, X_test_dropped
        del X_train_final_for_scaler
        gc.collect()

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Model and Evaluate ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "W6/models/enhanced_multitask_nn_best.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        # Determine model parameters (need to match training)
        input_size = X_test_scaled.shape[1]
        hidden_size1 = 256 # Must match train12.py
        hidden_size2 = 128 # Must match train12.py
        dropout_rate = 0.4 # Must match train12.py

        # Initialize model structure
        model = EnhancedMultiTaskNN(input_size, hidden_size1, hidden_size2, num_attack_classes, num_risk_classes, dropout_rate=dropout_rate)

        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")

        # Create DataLoader for the test set
        test_dataset = UNSWDataset(X_test_scaled, y_test_attack_np, y_test_severity_np)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0) # Use num_workers=0 if issues arise

        # Evaluate the model
        evaluate_model(model, test_loader, device, attack_class_names, []) # Pass empty list for risk names, generate in function

    except Exception as e:
        print(f"Error during model loading or evaluation: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nEvaluation script finished in {(time.time() - start_time_eval):.2f} seconds.")

if __name__ == "__main__":
    main()