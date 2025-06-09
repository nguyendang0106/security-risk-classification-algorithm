import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier # Needed for RFE definition
from sklearn.feature_selection import RFE
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ===================== Model and Dataset Definitions (Must match train4.py) =====================

class UNSWDataset(Dataset):
    # This might not be strictly needed if we pass tensors directly, but good practice
    def __init__(self, x, y_attack, y_risk):
        # Ensure x is a tensor
        if isinstance(x, pd.DataFrame):
            self.x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
             self.x = torch.tensor(x, dtype=torch.float32)
        else: # Assume it's already a tensor
             self.x = x

        # Ensure y are tensors
        if isinstance(y_attack, pd.Series):
            self.y_attack = torch.tensor(y_attack.values, dtype=torch.long)
        elif isinstance(y_attack, np.ndarray):
             self.y_attack = torch.tensor(y_attack, dtype=torch.long)
        else: # Assume it's already a tensor
             self.y_attack = y_attack

        if isinstance(y_risk, pd.Series):
            self.y_risk = torch.tensor(y_risk.values, dtype=torch.long)
        elif isinstance(y_risk, np.ndarray):
             self.y_risk = torch.tensor(y_risk, dtype=torch.long)
        else: # Assume it's already a tensor
             self.y_risk = y_risk


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_attack[idx], self.y_risk[idx]

class MultiTaskNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk):
        super(MultiTaskNeuralNet, self).__init__()
        self.shared_l1 = nn.Linear(input_size, hidden_size)
        self.shared_l2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.attack_out = nn.Linear(hidden_size_2, num_classes_attack)
        self.risk_out = nn.Linear(hidden_size_2, num_classes_risk) # Renamed from severity for clarity

    def forward(self, x):
        x = self.relu(self.shared_l1(x))
        x = self.relu(self.shared_l2(x))
        return self.attack_out(x), self.risk_out(x)

# ===================== Evaluation Function =====================

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
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

def evaluate_model(model, test_loader, device, attack_class_names, risk_class_names, model_desc):
    """Evaluates the multi-task model on the test set."""
    model.to(device)
    model.eval()

    all_y_attack_true = []
    all_y_risk_true = []
    all_y_attack_pred = []
    all_y_risk_pred = []

    print(f"\n--- Evaluating Model: {model_desc} ---")
    with torch.no_grad():
        for x, y_attack, y_risk in test_loader:
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
    # Ensure labels match the range of classes
    attack_labels = np.arange(len(attack_class_names))
    print(classification_report(all_y_attack_true, all_y_attack_pred, labels=attack_labels, target_names=attack_class_names, zero_division=0))
    plot_confusion_matrix_eval(all_y_attack_true, all_y_attack_pred, attack_class_names,
                               f'Attack Confusion Matrix ({model_desc})',
                               f'W6/results/attack_cm_{model_desc.replace(" ", "_").lower()}.png')

    # --- Risk/Severity Classification Metrics ---
    print("\n--- Risk/Severity Classification Results ---")
    risk_acc = accuracy_score(all_y_risk_true, all_y_risk_pred)
    risk_f1_weighted = f1_score(all_y_risk_true, all_y_risk_pred, average='weighted', zero_division=0)
    risk_f1_macro = f1_score(all_y_risk_true, all_y_risk_pred, average='macro', zero_division=0)
    print(f"Accuracy: {risk_acc:.4f}")
    print(f"F1 Score (Weighted): {risk_f1_weighted:.4f}")
    print(f"F1 Score (Macro): {risk_f1_macro:.4f}")
    print("Classification Report:")
    # Ensure labels match the range of classes
    risk_labels = np.arange(len(risk_class_names))
    print(classification_report(all_y_risk_true, all_y_risk_pred, labels=risk_labels, target_names=risk_class_names, zero_division=0))
    plot_confusion_matrix_eval(all_y_risk_true, all_y_risk_pred, risk_class_names,
                               f'Risk/Severity Confusion Matrix ({model_desc})',
                               f'W6/results/risk_cm_{model_desc.replace(" ", "_").lower()}.png')

    print("-" * 50)


# ===================== Main Evaluation Script =====================
def main():
    print("Starting evaluation script...")
    start_time = time.time()

    # Ensure results directory exists
    os.makedirs('W6/results', exist_ok=True)

    # --- 1. Load and Preprocess Data (Replicate train4.py steps) ---
    print("Loading and preprocessing data...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    try:
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)

        # Severity mapping (must match train4.py)
        severity_map = {
            "Normal": 0, "Analysis": 2, "Backdoor": 3, "DoS": 3, "Exploits": 4,
            "Fuzzers": 1, "Generic": 2, "Reconnaissance": 1, "Shellcode": 4, "Worms": 4
        }
        combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)

        # Label Encoding (Attack and Categorical Features)
        le_attack = LabelEncoder()
        combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
        attack_class_names = list(le_attack.classes_) # Store class names
        num_attack_classes = len(attack_class_names)

        # Create risk class names based on the map values
        num_risk_classes = max(severity_map.values()) + 1
        risk_class_names = [f"Risk_{i}" for i in range(num_risk_classes)] # Generic names

        combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
        combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
        combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

        # PCA (exactly as in train4.py)
        # FIX: Use numeric_only=True for std() and corr()
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat_encoded')['attack_cat_encoded'].nsmallest(7).index)
        exclude = list(set(lowCORR + lowSTD))
        # Ensure target columns and original attack_cat are not dropped if present in 'exclude'
        # (This loop might not be strictly necessary now but doesn't hurt)
        for col in ['attack_cat', 'attack_cat_encoded', 'severity_encoded']:
             if col in exclude:
                 exclude.remove(col)

        pca = PCA(n_components=3)
        # Fit PCA only on the features to be reduced
        # Ensure the columns in 'exclude' actually exist in combined_data before selecting
        valid_exclude = [col for col in exclude if col in combined_data.columns]
        pca_features_df = combined_data[valid_exclude] # Use valid_exclude
        # Handle potential NaNs before PCA
        pca_features_df = pca_features_df.fillna(pca_features_df.mean())
        dim_reduct = pca.fit_transform(pca_features_df)

        # Drop valid_exclude columns and original attack_cat
        combined_data_pca = combined_data.drop(valid_exclude + ['attack_cat'], axis=1)
        dim_reduction = pd.DataFrame(dim_reduct, columns=['PCA1', 'PCA2', 'PCA3'], index=combined_data.index)
        combined_data_pca = combined_data_pca.join(dim_reduction)

        # Prepare X and Y
        # Ensure target columns exist before dropping
        cols_to_drop_pca = ['attack_cat_encoded', 'severity_encoded']
        cols_to_drop_pca = [col for col in cols_to_drop_pca if col in combined_data_pca.columns]
        data_x = combined_data_pca.drop(columns=cols_to_drop_pca)
        data_y_attack = combined_data_pca['attack_cat_encoded']
        data_y_severity = combined_data_pca['severity_encoded']

        # Scaling (MinMaxScaler as used in train4.py before splitting for NN)
        # Note: Ideally, scaler should be fit ONLY on training data.
        # To perfectly replicate train4.py's potential data leakage for evaluation:
        scaler_all = MinMaxScaler()
        data_x_scaled_all = pd.DataFrame(scaler_all.fit_transform(data_x), columns=data_x.columns)

        # Split data (using the same random_state as train4.py)
        # We only need the test set for evaluation
        _, X_test_all, _, y_test_attack, _, y_test_severity = train_test_split(
            data_x_scaled_all, data_y_attack, data_y_severity, test_size=0.20, random_state=42
        )
        X_test_all.columns = X_test_all.columns.astype(str) # Ensure string column names

        # --- Prepare data for the "Selected Features" model ---
        # Re-fit RFE on the *original* training split used in train4.py to get the same features
        # This requires splitting the *unscaled* data first
        X_train_orig, _, y_train_attack_orig, _ = train_test_split(
            data_x, data_y_attack, test_size=0.20, random_state=42
        )
        # Scale the original training data for RFE fitting
        scaler_train = MinMaxScaler()
        X_train_orig_scaled = scaler_train.fit_transform(X_train_orig)
        X_train_orig_scaled_df = pd.DataFrame(X_train_orig_scaled, columns=data_x.columns)

        print("Running RFE to determine selected features (as in train4.py)...")
        rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10).fit(X_train_orig_scaled_df, y_train_attack_orig)
        selected_indices = np.where(rfe.support_==True)[0]
        selected_features = list(data_x.columns[selected_indices])
        print("Selected features identified:", selected_features)

        # Select features from the *already scaled* full dataset
        data_x_selected = data_x_scaled_all[selected_features]

        # Split the selected feature data
        _, X_test_selected, _, _, _, _ = train_test_split( # Only need X_test_selected
            data_x_selected, data_y_attack, data_y_severity, test_size=0.20, random_state=42
        )
        X_test_selected.columns = X_test_selected.columns.astype(str) # Ensure string column names

        # Clean up intermediate dataframes
        del train, test, combined_data, combined_data_pca, data_x, data_x_scaled_all, data_x_selected
        del X_train_orig, y_train_attack_orig, X_train_orig_scaled_df
        gc.collect()

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Models and Evaluate ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Evaluate Model 1: All Features ---
    model_path_all = "W6/models/multitask_nn_all_features.pth"
    if os.path.exists(model_path_all):
        try:
            input_size_all = X_test_all.shape[1]
            model_all = MultiTaskNeuralNet(input_size_all, 128, 64, num_attack_classes, num_risk_classes)
            model_all.load_state_dict(torch.load(model_path_all, map_location=device))

            # Create DataLoader for test set (all features)
            test_dataset_all = UNSWDataset(X_test_all, y_test_attack, y_test_severity)
            test_loader_all = DataLoader(test_dataset_all, batch_size=128, shuffle=False)

            evaluate_model(model_all, test_loader_all, device, attack_class_names, risk_class_names, "All Features")
            del model_all, test_dataset_all, test_loader_all # Memory cleanup
            gc.collect()
        except Exception as e:
            print(f"Error evaluating 'All Features' model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file not found: {model_path_all}")

    # --- Evaluate Model 2: Selected Features ---
    model_path_selected = "W6/models/multitask_nn_selected_features.pth"
    if os.path.exists(model_path_selected):
        try:
            input_size_selected = X_test_selected.shape[1]
            # Ensure input size matches the number of selected features
            if input_size_selected != len(selected_features):
                 print(f"Warning: Mismatch in selected feature count ({input_size_selected}) vs expected ({len(selected_features)}). Check RFE/preprocessing.")

            model_selected = MultiTaskNeuralNet(input_size_selected, 128, 64, num_attack_classes, num_risk_classes)
            model_selected.load_state_dict(torch.load(model_path_selected, map_location=device))

            # Create DataLoader for test set (selected features)
            test_dataset_selected = UNSWDataset(X_test_selected, y_test_attack, y_test_severity)
            test_loader_selected = DataLoader(test_dataset_selected, batch_size=128, shuffle=False)

            evaluate_model(model_selected, test_loader_selected, device, attack_class_names, risk_class_names, "Selected Features (RFE)")
            del model_selected, test_dataset_selected, test_loader_selected # Memory cleanup
            gc.collect()
        except Exception as e:
            print(f"Error evaluating 'Selected Features' model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file not found: {model_path_selected}")


    print(f"\nEvaluation script finished in {(time.time() - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()