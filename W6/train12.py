import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for Focal Loss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # Changed to StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os
import collections
from tabulate import tabulate
import time
import gc
from tqdm import tqdm # For progress bar

# ===================== Focal Loss Implementation =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Can be a float or a tensor of weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ===================== Dataset Class (Modified for tensors) =====================
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

# ===================== Enhanced Multi-Task Neural Network =====================
class EnhancedMultiTaskNN(nn.Module):
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
        x = self.bn_input(x) # Apply BN to input
        x = self.relu(self.bn1(self.shared_l1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.shared_l2(x)))
        x = self.dropout(x)
        out_attack = self.attack_out(x)
        out_risk = self.risk_out(x)
        return out_attack, out_risk

# ===================== Improved Training Function =====================
def train_enhanced_nn(model, train_loader, val_loader, device, num_classes_attack, num_classes_risk,
                      epochs=50, lr=0.001, weight_decay=1e-5, patience=10,
                      loss_weight_attack=0.6, gamma_focal=2.0, model_save_path="W6/models/enhanced_multitask_nn.pth"):

    # Use Focal Loss
    # You might want to calculate class weights for alpha if imbalance is severe
    criterion_attack = FocalLoss(gamma=gamma_focal, reduction='mean')
    criterion_risk = FocalLoss(gamma=gamma_focal, reduction='mean') # Can use different gamma/alpha if needed

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2, verbose=True)

    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc_attack': [], 'val_acc_attack': [], 'train_acc_risk': [], 'val_acc_risk': []}

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_attack = 0
        correct_risk = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [T]", leave=False)

        for x, y_attack, y_risk in progress_bar:
            x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)
            optimizer.zero_grad()

            out_attack, out_risk = model(x)
            loss_attack = criterion_attack(out_attack, y_attack)
            loss_risk = criterion_risk(out_risk, y_risk)
            # Weighted combination of losses
            loss = loss_weight_attack * loss_attack + (1 - loss_weight_attack) * loss_risk

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0) # Accumulate weighted loss
            _, pred_attack = torch.max(out_attack, 1)
            _, pred_risk = torch.max(out_risk, 1)
            correct_attack += (pred_attack == y_attack).sum().item()
            correct_risk += (pred_risk == y_risk).sum().item()
            total += y_attack.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / total
        train_acc_attack = correct_attack / total
        train_acc_risk = correct_risk / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc_attack'].append(train_acc_attack)
        history['train_acc_risk'].append(train_acc_risk)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct_attack = 0
        val_correct_risk = 0
        val_total = 0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [V]", leave=False)

        with torch.no_grad():
            for x, y_attack, y_risk in val_progress_bar:
                x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)
                out_attack, out_risk = model(x)
                loss_attack = criterion_attack(out_attack, y_attack)
                loss_risk = criterion_risk(out_risk, y_risk)
                loss = loss_weight_attack * loss_attack + (1 - loss_weight_attack) * loss_risk

                val_loss += loss.item() * x.size(0)
                _, pred_attack = torch.max(out_attack, 1)
                _, pred_risk = torch.max(out_risk, 1)
                val_correct_attack += (pred_attack == y_attack).sum().item()
                val_correct_risk += (pred_risk == y_risk).sum().item()
                val_total += y_attack.size(0)
                val_progress_bar.set_postfix({'val_loss': loss.item()})

        avg_val_loss = val_loss / val_total
        val_acc_attack = val_correct_attack / val_total
        val_acc_risk = val_correct_risk / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc_attack'].append(val_acc_attack)
        history['val_acc_risk'].append(val_acc_risk)

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc Att: {train_acc_attack:.4f}, Val Acc Att: {val_acc_attack:.4f} | "
              f"Train Acc Risk: {train_acc_risk:.4f}, Val Acc Risk: {val_acc_risk:.4f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    # Load the best model state
    print(f"Loading best model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    return model, history


def main():
    print("Loading datasets...")
    start_time_main = time.time()
    # --- Data Loading (same as before) ---
    try:
        if not os.path.exists('W6/data'):
            os.makedirs('W6/data', exist_ok=True) # Ensure data dir exists
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    try:
        # --- Basic Preprocessing (Label Encoding, Severity Mapping) ---
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
        print(f"Combined data shape: {combined_data.shape}")

        severity_map = {
            "Normal": 0, "Analysis": 2, "Backdoor": 3, "DoS": 3, "Exploits": 4,
            "Fuzzers": 1, "Generic": 2, "Reconnaissance": 1, "Shellcode": 4, "Worms": 4
        }
        combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)

        le_attack = LabelEncoder()
        vector = combined_data['attack_cat'] # Keep original names for mapping later
        combined_data['attack_cat_encoded'] = le_attack.fit_transform(vector)
        attack_mapping = {i: cls_name for i, cls_name in enumerate(le_attack.classes_)} # Store mapping
        num_attack_classes = len(attack_mapping)
        num_risk_classes = combined_data['severity_encoded'].nunique() # Or max + 1

        combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
        combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
        combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

        # --- Feature Analysis for PCA (Numeric Only) ---
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat_encoded')['attack_cat_encoded'].nsmallest(7).index)
        exclude_pca_cols = list(set(lowCORR + lowSTD))
        # Ensure target/ID columns are not included in PCA features
        for col in ['attack_cat', 'attack_cat_encoded', 'severity_encoded']:
             if col in exclude_pca_cols:
                 exclude_pca_cols.remove(col)
        print('Columns selected for PCA reduction:', exclude_pca_cols)

        # --- Prepare Data (Drop original attack_cat) ---
        data_x_full = combined_data.drop(['attack_cat', 'attack_cat_encoded', 'severity_encoded'], axis=1)
        data_y_attack = combined_data['attack_cat_encoded']
        data_y_severity = combined_data['severity_encoded']

        # --- Train/Validation/Test Split (BEFORE Scaling/PCA) ---
        print("Splitting data into Train/Validation/Test sets (60/20/20)...")
        X_train_val, X_test, y_train_val_attack, y_test_attack, y_train_val_severity, y_test_severity = train_test_split(
            data_x_full, data_y_attack, data_y_severity, test_size=0.20, random_state=42, stratify=data_y_attack
        )
        X_train, X_val, y_train_attack, y_val_attack, y_train_severity, y_val_severity = train_test_split(
            X_train_val, y_train_val_attack, y_train_val_severity, test_size=0.25, random_state=42, stratify=y_train_val_attack # 0.25 * 0.8 = 0.2
        )

        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

        # --- Apply PCA (Fit on Train, Transform Train/Val/Test) ---
        print("Applying PCA...")
        pca = PCA(n_components=3)
        valid_exclude_train = [col for col in exclude_pca_cols if col in X_train.columns]
         # --- Create copies to avoid modifying original split data during PCA prep ---
        X_train_pca_input = X_train[valid_exclude_train].copy()
        X_val_pca_input = X_val[valid_exclude_train].copy()
        X_test_pca_input = X_test[valid_exclude_train].copy()

        # Handle NaNs using train set's mean
        train_means = X_train_pca_input.mean()
        X_train_pca_input.fillna(train_means, inplace=True)
        X_val_pca_input.fillna(train_means, inplace=True)
        X_test_pca_input.fillna(train_means, inplace=True)


        # Fit PCA ONLY on training data's selected columns
        pca.fit(X_train_pca_input)
        print("PCA Explained variance ratio:", sum(pca.explained_variance_ratio_))

        # Transform train, val, test
        train_pca_res = pca.transform(X_train_pca_input)
        val_pca_res = pca.transform(X_val_pca_input)
        test_pca_res = pca.transform(X_test_pca_input)

        # Create PCA dataframes with correct indices
        pca_cols = ['PCA1', 'PCA2', 'PCA3']
        train_pca_df = pd.DataFrame(train_pca_res, columns=pca_cols, index=X_train.index)
        val_pca_df = pd.DataFrame(val_pca_res, columns=pca_cols, index=X_val.index)
        test_pca_df = pd.DataFrame(test_pca_res, columns=pca_cols, index=X_test.index)

        # --- Prepare final feature sets ---
        # Drop original PCA columns
        X_train_dropped = X_train.drop(columns=valid_exclude_train)
        X_val_dropped = X_val.drop(columns=valid_exclude_train)
        X_test_dropped = X_test.drop(columns=valid_exclude_train)

        # --- Explicitly check index equality before combining ---
        print("\n--- Verifying Index Equality Before Combining ---")
        print(f"Train indices equal: {X_train_dropped.index.equals(train_pca_df.index)}")
        print(f"Validation indices equal: {X_val_dropped.index.equals(val_pca_df.index)}")
        print(f"Test indices equal: {X_test_dropped.index.equals(test_pca_df.index)}")
        assert X_train_dropped.index.equals(train_pca_df.index), "Train indices mismatch before combining!"
        assert X_val_dropped.index.equals(val_pca_df.index), "Validation indices mismatch before combining!"
        assert X_test_dropped.index.equals(test_pca_df.index), "Test indices mismatch before combining!"
        print("--- Index Equality Verification Passed ---")

        # Combine using pd.concat along columns (axis=1)
        print("Combining features using pd.concat...")
        X_train_final = pd.concat([X_train_dropped, train_pca_df], axis=1)
        X_val_final = pd.concat([X_val_dropped, val_pca_df], axis=1)
        X_test_final = pd.concat([X_test_dropped, test_pca_df], axis=1)

        # --- Verify shapes after combining ---
        print("\n--- Verifying Shapes After Combining ---")
        print(f"X_train_final shape: {X_train_final.shape}")
        print(f"y_train_attack shape: {y_train_attack.shape}")
        # Add print shapes right before assert
        print(f"Comparing: X_train_final.shape[0]={X_train_final.shape[0]} vs y_train_attack.shape[0]={y_train_attack.shape[0]}")
        assert X_train_final.shape[0] == y_train_attack.shape[0], "Train shapes mismatch after combining!"
        print("--- Combining Shape Verification Passed ---")

        print(f"Shape after PCA: Train={X_train_final.shape}, Val={X_val_final.shape}, Test={X_test_final.shape}")

        # --- Scaling (Fit on Train, Transform Train/Val/Test) ---
        print("Applying StandardScaler...")
        scaler = StandardScaler()
        # Use X_*_final for scaling
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val_final)
        X_test_scaled = scaler.transform(X_test_final)

       # Convert target Series to numpy arrays
        y_train_attack_np = y_train_attack.values
        y_val_attack_np = y_val_attack.values
        y_test_attack_np = y_test_attack.values
        y_train_severity_np = y_train_severity.values
        y_val_severity_np = y_val_severity.values
        y_test_severity_np = y_test_severity.values

        # --- Add Shape Verification ---
        print("\n--- Verifying Shapes Before Dataset Creation ---")
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"y_train_attack_np shape: {y_train_attack_np.shape}")
        print(f"y_train_severity_np shape: {y_train_severity_np.shape}")
        assert X_train_scaled.shape[0] == y_train_attack_np.shape[0] == y_train_severity_np.shape[0], "Train data shapes mismatch!"

        print(f"\nX_val_scaled shape: {X_val_scaled.shape}")
        print(f"y_val_attack shape: {y_val_attack.shape}")
        print(f"y_val_severity shape: {y_val_severity.shape}")
        assert X_val_scaled.shape[0] == y_val_attack.shape[0] == y_val_severity.shape[0], "Validation data shapes mismatch!"

        print(f"\nX_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_test_attack shape: {y_test_attack.shape}")
        print(f"y_test_severity shape: {y_test_severity.shape}")
        assert X_test_scaled.shape[0] == y_test_attack.shape[0] == y_test_severity.shape[0], "Test data shapes mismatch!"
        print("--- Shape Verification Passed ---")
        # --- End Shape Verification ---


        # Clean up memory
        del combined_data, data_x_full, X_train_val, y_train_val_attack, y_train_val_severity
        del X_train, X_val, X_test # Use scaled versions from now on
        gc.collect()

        # --- Prepare DataLoaders ---
        # Use the numpy versions of targets
        train_dataset = UNSWDataset(X_train_scaled, y_train_attack_np, y_train_severity_np)
        val_dataset = UNSWDataset(X_val_scaled, y_val_attack_np, y_val_severity_np)
        test_dataset = UNSWDataset(X_test_scaled, y_test_attack_np, y_test_severity_np) # For final evaluation

        batch_size = 256 # Increased batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0) # Also set for test_loader if used similarly

        # --- Initialize and Train Model ---
        print("\n================== Training Enhanced Model ==================")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        input_size = X_train_scaled.shape[1]
        hidden_size1 = 256 # Increased hidden size
        hidden_size2 = 128
        dropout_rate = 0.4 # Adjusted dropout

        model = EnhancedMultiTaskNN(input_size, hidden_size1, hidden_size2, num_attack_classes, num_risk_classes, dropout_rate=dropout_rate)

        model_save_path = "W6/models/enhanced_multitask_nn_best.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists

        # Train the model
        trained_model, history = train_enhanced_nn(
            model, train_loader, val_loader, device, num_attack_classes, num_risk_classes,
            epochs=40, # Increased epochs, early stopping will handle it
            lr=0.0005, # Adjusted learning rate
            weight_decay=1e-4,
            patience=15, # Increased patience
            loss_weight_attack=0.6, # Weight attack loss slightly higher
            gamma_focal=2.0,
            model_save_path=model_save_path
        )

        # --- Final Evaluation on Test Set (using the best model loaded in train function) ---
        # (You would typically call an evaluation function here similar to testtrain4.py)
        print("\n--- Evaluating Best Model on TEST Set ---")
        # Re-use the evaluation logic from testtrain4.py (evaluate_model function)
        # Need to adapt it slightly if necessary or copy it here.
        # For brevity, let's just print basic test accuracy here.
        trained_model.eval()
        test_correct_attack = 0
        test_correct_risk = 0
        test_total = 0
        with torch.no_grad():
            for x, y_attack, y_risk in test_loader:
                x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)
                out_attack, out_risk = trained_model(x)
                _, pred_attack = torch.max(out_attack, 1)
                _, pred_risk = torch.max(out_risk, 1)
                test_correct_attack += (pred_attack == y_attack).sum().item()
                test_correct_risk += (pred_risk == y_risk).sum().item()
                test_total += y_attack.size(0)

        test_acc_attack = test_correct_attack / test_total
        test_acc_risk = test_correct_risk / test_total
        print(f"Test Set -> Attack Accuracy: {test_acc_attack:.4f}, Risk Accuracy: {test_acc_risk:.4f}")
        # TODO: Add full evaluation report (precision, recall, F1, confusion matrix) using the test_loader

        print(f"\nScript finished in {(time.time() - start_time_main)/60:.2f} minutes.")

    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()