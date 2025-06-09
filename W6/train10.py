import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from tabulate import tabulate
import warnings
from tqdm import tqdm
import time
import gc

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.combine import SMOTETomek


# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure models directory exists
os.makedirs('W6/models', exist_ok=True)

# Risk level mapping
RISK_MAP = {
    "Normal": "UNKNOWN",
    "Analysis": "LOW",
    "Backdoor": "HIGH",
    "DoS": "HIGH",
    "Exploits": "CRITICAL",
    "Fuzzers": "LOW",
    "Generic": "MEDIUM",
    "Reconnaissance": "LOW",
    "Shellcode": "CRITICAL",
    "Worms": "CRITICAL"
}

# Focal Loss Implementation with label smoothing
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(1)
        device = inputs.device
        
        # Create one-hot encoding of targets
        one_hot = torch.zeros(targets.size(0), n_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        
        # Calculate CE loss with smoothed targets
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(smooth_one_hot * log_probs, dim=1)
        
        # Apply focal scaling
        pt = torch.exp(-loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting (optional)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha[targets]
                focal_weight = alpha * focal_weight
            else:
                focal_weight = self.alpha * focal_weight

        focal_loss = focal_weight * loss
                
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# Enhanced Neural Network with ResNet-like connections and improved architecture
class EnhancedDualNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes_attack, num_classes_risk, dropout_rate=0.3):
        super(EnhancedDualNN, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn_input = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout = nn.Dropout(dropout_rate)

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        # Add more capacity with deeper network
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            
            # Add residual connection if shapes match, otherwise add projection
            if hidden_sizes[i] == hidden_sizes[i+1]:
                self.residual_layers.append(None)  # Identity mapping
            else:
                self.residual_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Attack classification branch - deeper and wider
        self.attack_hidden1 = nn.Linear(hidden_sizes[-1], 256)
        self.attack_bn1 = nn.BatchNorm1d(256)
        self.attack_hidden2 = nn.Linear(256, 128)
        self.attack_bn2 = nn.BatchNorm1d(128)
        self.attack_hidden3 = nn.Linear(128, 64)  # Extra layer for capacity
        self.attack_bn3 = nn.BatchNorm1d(64)
        self.attack_head = nn.Linear(64, num_classes_attack)

        # Risk classification branch - separate layers
        self.risk_hidden1 = nn.Linear(hidden_sizes[-1], 128)
        self.risk_bn1 = nn.BatchNorm1d(128)
        self.risk_hidden2 = nn.Linear(128, 64)
        self.risk_bn2 = nn.BatchNorm1d(64)
        self.risk_head = nn.Linear(64, num_classes_risk)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.leaky_relu(self.bn_input(x))
        x = self.dropout(x)

        # Hidden layers with residual connections
        for i, (layer, bn, res) in enumerate(zip(self.hidden_layers, self.bn_layers, self.residual_layers)):
            identity = x  # Store input for residual connection
            x = layer(x)
            x = self.leaky_relu(bn(x))
            
            # Apply residual connection if shapes match
            if res is None:
                x = x + identity  # Direct addition
            else:
                x = x + res(identity)  # Projection needed
                
            x = self.dropout(x)

        # Attack classification pathway - deeper for better feature extraction
        attack = self.attack_hidden1(x)
        attack = self.leaky_relu(self.attack_bn1(attack))
        attack = self.dropout(attack)
        
        attack = self.attack_hidden2(attack)
        attack = self.leaky_relu(self.attack_bn2(attack))
        attack = self.dropout(attack)
        
        attack = self.attack_hidden3(attack)  # Extra layer
        attack = self.leaky_relu(self.attack_bn3(attack))
        attack = self.dropout(attack)
        
        attack_output = self.attack_head(attack)

        # Risk classification pathway
        risk = self.risk_hidden1(x)
        risk = self.leaky_relu(self.risk_bn1(risk))
        risk = self.dropout(risk)
        
        risk = self.risk_hidden2(risk)
        risk = self.leaky_relu(self.risk_bn2(risk))
        risk = self.dropout(risk)
        
        risk_output = self.risk_head(risk)

        return attack_output, risk_output

# Function to plot confusion matrix with correct labels
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save confusion matrix"""
    # Ensure labels are consistent for confusion matrix calculation
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate normalized confusion matrix for better visualization
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # Create two subplots - one for raw counts, one for percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"{title} (Counts)")
    plt.colorbar(im1, ax=ax1)

    # Plot normalized percentages
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f"{title} (Normalized)")
    plt.colorbar(im2, ax=ax2)

    # Add labels to both plots
    for ax in [ax1, ax2]:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    # Add text annotations
    thresh1 = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh1 else "black")

    # Add text annotations for percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, f"{cm_normalized[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

# Apply SMOTE and feature selection for better class balance
def prepare_balanced_data(X_train, y_train_attack, y_train_risk, k_features=100):
    """Apply SMOTETomek to balance classes and feature selection"""
    print("\nApplying SMOTETomek for attack class balancing and cleaning...")

    # Create a unique identifier for each sample
    n_samples = X_train.shape[0]
    ids = np.arange(n_samples).reshape(-1, 1)
    X_train_with_ids = np.hstack((X_train, ids))

    # Use SMOTETomek for attack classification
    # n_jobs=-1 uses all available CPU cores
    smt = SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=-1)
    X_train_attack_res_with_ids, y_train_attack_res = smt.fit_resample(X_train_with_ids, y_train_attack)

    # Extract IDs and features
    X_train_attack_res = X_train_attack_res_with_ids[:, :-1]
    res_ids = X_train_attack_res_with_ids[:, -1].astype(int)
    is_original = res_ids < n_samples

    # Create aligned risk labels
    y_train_risk_res = np.zeros_like(y_train_attack_res)
    for i, idx in enumerate(res_ids):
        if is_original[i]:
            y_train_risk_res[i] = y_train_risk[idx]

    # Map attack to risk for synthetic samples
    attack_to_risk_map = {}
    for a_label, r_label in zip(y_train_attack, y_train_risk):
        if a_label not in attack_to_risk_map:
            attack_to_risk_map[a_label] = collections.Counter()
        attack_to_risk_map[a_label][r_label] += 1

    synthetic_indices = np.where(~is_original)[0]
    for i in synthetic_indices:
        attack_label = y_train_attack_res[i]
        if attack_label in attack_to_risk_map:
            # Assign most common risk label
            most_common_risk = attack_to_risk_map[attack_label].most_common(1)[0][0]
            y_train_risk_res[i] = most_common_risk
        else:
            y_train_risk_res[i] = 0 # Fallback

    print(f"Original training set shape: {X_train.shape}")
    print(f"Balanced+Cleaned training set shape: {X_train_attack_res.shape}")

    # Feature selection using SelectKBest
    print(f"Applying SelectKBest to select top {k_features} features...")
    # Ensure k_features is not more than the number of available features
    k_features = min(k_features, X_train_attack_res.shape[1])
    selector = SelectKBest(f_classif, k=k_features)
    # FIX: Use the balanced data for fitting the selector
    X_train_selected = selector.fit_transform(X_train_attack_res, y_train_attack_res)
    selected_indices = selector.get_support(indices=True) # Get indices of selected features
    print(f"Selected {X_train_selected.shape[1]} features.")

    # FIX: Return the selected training data, not the full balanced data
    return X_train_selected, y_train_attack_res, y_train_risk_res, selected_indices # Return indices

# Function to train the enhanced neural network using Focal Loss and SMOTE
def train_enhanced_neural_network(X_train, y_train_attack, y_train_risk,
                                  X_valid, y_valid_attack, y_valid_risk, # Validation set added
                                  X_test, y_test_attack, y_test_risk,
                                  attack_mapping, risk_mapping, selected_indices): # Indices added    """Train an enhanced neural network for dual classification using advanced techniques"""
    print("\n=== Training Enhanced Dual Neural Network with Advanced Techniques ===")
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare balanced data using SMOTETomek and feature selection
    # Note: k_features can be tuned
    X_train_balanced, y_train_attack_balanced, y_train_risk_balanced, selected_indices = prepare_balanced_data(
        X_train, y_train_attack, y_train_risk, k_features=100
    )

    # Apply feature selection to validation and test sets
    X_valid_selected = X_valid[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    # Hyperparameters - Tuned
    input_size = X_train_balanced.shape[1] # Should match k_features
    hidden_sizes = [512, 256, 128] # Slightly simplified hidden layers
    num_classes_attack = len(attack_mapping)
    num_classes_risk = len(risk_mapping)
    num_epochs = 30 # Increased epochs
    batch_size = 512
    learning_rate = 0.0002 # Adjusted LR
    weight_decay = 5e-5 # Adjusted weight decay
    gamma_focal = 1.0 # Reduced gamma
    patience = 10 # Increased patience
    label_smoothing = 0.1

    print(f"Network architecture: input={input_size}, hidden={hidden_sizes}, "
          f"attack_classes={num_classes_attack}, risk_classes={num_classes_risk}")

    # Initialize model
    model = EnhancedDualNN( # Assuming EnhancedDualNN is defined as before
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes_attack=num_classes_attack,
        num_classes_risk=num_classes_risk,
        dropout_rate=0.3
    ).to(device)

    # Loss functions
    criterion_attack = FocalLoss(gamma=gamma_focal, label_smoothing=label_smoothing)
    criterion_risk = FocalLoss(gamma=gamma_focal, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7) # Adjusted T_max

    # Convert data to Tensors
    X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32).to(device)
    y_train_attack_tensor = torch.tensor(y_train_attack_balanced, dtype=torch.long).to(device)
    y_train_risk_tensor = torch.tensor(y_train_risk_balanced, dtype=torch.long).to(device)

    X_valid_tensor = torch.tensor(X_valid_selected, dtype=torch.float32).to(device) # Use selected features
    y_valid_attack_tensor = torch.tensor(y_valid_attack, dtype=torch.long).to(device)
    y_valid_risk_tensor = torch.tensor(y_valid_risk, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device) # Use selected features
    y_test_attack_tensor = torch.tensor(y_test_attack, dtype=torch.long).to(device)
    y_test_risk_tensor = torch.tensor(y_test_risk, dtype=torch.long).to(device)


    # DataLoaders
    dataset = TensorDataset(X_train_tensor, y_train_attack_tensor, y_train_risk_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training loop
    best_combined_f1 = 0
    no_improve_epochs = 0
    best_val_loss = float('inf') # Track best validation loss

    print(f"Starting training for {num_epochs} epochs (with early stopping)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_x, batch_y_attack, batch_y_risk in progress_bar:
            batch_x, batch_y_attack, batch_y_risk = batch_x.to(device), batch_y_attack.to(device), batch_y_risk.to(device)

            outputs_attack, outputs_risk = model(batch_x)
            loss_attack = criterion_attack(outputs_attack, batch_y_attack)
            loss_risk = criterion_risk(outputs_risk, batch_y_risk)
            loss = 0.7 * loss_attack + 0.3 * loss_risk # Fixed weighting

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        scheduler.step()

        # Validation phase
        model.eval()
        val_total_loss = 0
        val_batch_count = 0
        all_val_attack_preds = []
        all_val_risk_preds = []
        
        # Evaluation
        with torch.no_grad():
             # Manual iteration over validation set if it's small, or use DataLoader
            val_outputs_attack, val_outputs_risk = model(X_valid_tensor)
            val_loss_attack = criterion_attack(val_outputs_attack, y_valid_attack_tensor)
            val_loss_risk = criterion_risk(val_outputs_risk, y_valid_risk_tensor)
            val_loss = 0.7 * val_loss_attack + 0.3 * val_loss_risk

            _, val_attack_preds = torch.max(val_outputs_attack, 1)
            _, val_risk_preds = torch.max(val_outputs_risk, 1)

            all_val_attack_preds.extend(val_attack_preds.cpu().numpy())
            all_val_risk_preds.extend(val_risk_preds.cpu().numpy())

        # Calculate validation metrics
        val_attack_acc = accuracy_score(y_valid_attack, all_val_attack_preds)
        val_risk_acc = accuracy_score(y_valid_risk, all_val_risk_preds)
        val_attack_f1 = f1_score(y_valid_attack, all_val_attack_preds, average='weighted', zero_division=0)
        val_risk_f1 = f1_score(y_valid_risk, all_val_risk_preds, average='weighted', zero_division=0)
        val_combined_f1 = 0.7 * val_attack_f1 + 0.3 * val_risk_f1

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, "
              f"Val Attack: {val_attack_acc*100:.2f}% (F1={val_attack_f1:.4f}), "
              f"Val Risk: {val_risk_acc*100:.2f}% (F1={val_risk_f1:.4f})")

        # Early stopping based on validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_combined_f1 = val_combined_f1 # Store F1 corresponding to best val loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'W6/models/best_dual_nn_advanced_v2.pt')
            print(f"Validation loss improved ({best_val_loss:.4f}). Model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model based on validation loss
    print("\nLoading best model for final evaluation...")
    try:
        model.load_state_dict(torch.load('W6/models/best_dual_nn_advanced_v2.pt'))
    except FileNotFoundError:
        print("Warning: Best model checkpoint not found.")

    # Final evaluation
    print("\n=== Final Model Evaluation (Advanced Model) ===")
    model.eval()
    with torch.no_grad():
        outputs_attack, outputs_risk = model(X_test_tensor)
        _, attack_preds = torch.max(outputs_attack, 1)
        _, risk_preds = torch.max(outputs_risk, 1)

        attack_preds_np = attack_preds.cpu().numpy()
        risk_preds_np = risk_preds.cpu().numpy()
        y_test_attack_np = y_test_attack_tensor.cpu().numpy()
        y_test_risk_np = y_test_risk_tensor.cpu().numpy()

        final_attack_acc = accuracy_score(y_test_attack_np, attack_preds_np) * 100
        final_risk_acc = accuracy_score(y_test_risk_np, risk_preds_np) * 100
        final_attack_f1 = f1_score(y_test_attack_np, attack_preds_np, average='weighted', zero_division=0)
        final_risk_f1 = f1_score(y_test_risk_np, risk_preds_np, average='weighted', zero_division=0)

        print(f"Attack Classification Accuracy: {final_attack_acc:.2f}%")
        print(f"Attack Classification F1 Score: {final_attack_f1:.4f}")
        print(f"Risk Classification Accuracy: {final_risk_acc:.2f}%")
        print(f"Risk Classification F1 Score: {final_risk_f1:.4f}")

        # Classification reports
        print("\nAttack Classification Report:")
        unique_labels_attack = np.arange(num_classes_attack)
        attack_target_names = [attack_mapping.get(i, f"Class_{i}") for i in unique_labels_attack]
        print(classification_report(y_test_attack_np, attack_preds_np, 
                                   labels=unique_labels_attack, 
                                   target_names=attack_target_names, 
                                   zero_division=0))

        print("\nRisk Classification Report:")
        unique_labels_risk = np.arange(num_classes_risk)
        risk_target_names = [risk_mapping.get(i, f"Risk_{i}") for i in unique_labels_risk]
        print(classification_report(y_test_risk_np, risk_preds_np, 
                                   labels=unique_labels_risk, 
                                   target_names=risk_target_names, 
                                   zero_division=0))

    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")
    return model, final_attack_acc, final_risk_acc, final_attack_f1, final_risk_f1, attack_preds_np, risk_preds_np, selected_indices


def main():
    print("=== Network Intrusion Detection: Dual Classification with Advanced Techniques ===")
    overall_start_time = time.time()

    # Load data
    print("\nLoading datasets...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files 'UNSW_NB15_training-set.csv' and 'UNSW_NB15_testing-set.csv' exist in the 'W6/data/' directory.")
        return

    # Combine datasets
    combined_data = pd.concat([train, test], ignore_index=True).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")

    # Add risk level
    combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP).fillna('UNKNOWN')

    # Label encoding
    le_attack = LabelEncoder()
    le_risk = LabelEncoder()
    combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
    combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])

    # Mappings
    attack_mapping = {i: cat for i, cat in enumerate(le_attack.classes_)}
    risk_mapping = {i: level for i, level in enumerate(le_risk.classes_)}
    print("\nAttack category mapping:", attack_mapping)
    print("Risk level mapping:", risk_mapping)

    # Encode categorical features
    print("\nEncoding categorical features...")
    categorical_features = ['proto', 'service', 'state']
    for feature in categorical_features:
        if feature in combined_data.columns:
            combined_data[feature] = LabelEncoder().fit_transform(combined_data[feature].astype(str))
        else:
            print(f"Warning: Feature '{feature}' not found.")

    # Display class distributions
    print("\nAttack Type Distribution (Original):")
    print(tabulate(combined_data['attack_cat'].value_counts().reset_index().values.tolist(), 
                  headers=['Attack Type', 'Count']))
    print("\nRisk Level Distribution (Original):")
    print(tabulate(combined_data['risk_level'].value_counts().reset_index().values.tolist(), 
                  headers=['Risk Level', 'Count']))

    # Feature preparation
    print("\nPreparing features...")
    features_to_drop = ['attack_cat', 'risk_level', 'attack_cat_encoded', 'risk_level_encoded']
    data_x = combined_data.drop(columns=features_to_drop, errors='ignore')
    data_y_attack = combined_data['attack_cat_encoded']
    data_y_risk = combined_data['risk_level_encoded']

    non_numeric_cols = data_x.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"Warning: Non-numeric columns remaining: {list(non_numeric_cols)}. Dropping them.")
        data_x = data_x.drop(columns=non_numeric_cols)

    # Feature engineering - create derived features
    print("Adding engineered features...")
    # Example feature engineering - can be expanded
    data_x['bytes_per_packet'] = data_x['sbytes'] / (data_x['spkts'] + 1e-6)
    data_x['bytes_per_packet_received'] = data_x['dbytes'] / (data_x['dpkts'] + 1e-6)
    data_x['total_bytes'] = data_x['sbytes'] + data_x['dbytes']
    data_x['total_pkts'] = data_x['spkts'] + data_x['dpkts']
    for col in ['bytes_per_packet', 'bytes_per_packet_received', 'total_bytes', 'total_pkts']:
        q1 = data_x[col].quantile(0.01)
        q3 = data_x[col].quantile(0.99)
        data_x[col] = data_x[col].clip(q1, q3)

    
    # --- FIX: Scaling after split ---
    # Split data into train+validation and test FIRST
    print("Splitting data into train/validation/test sets...")
    X_train_val, X_test, y_train_val_attack, y_test_attack, y_train_val_risk, y_test_risk = train_test_split(
        data_x, data_y_attack, data_y_risk,
        test_size=0.20, # Test set size
        random_state=42,
        stratify=data_y_attack # Stratify based on the more complex task
    )

    # Split train+validation into train and validation
    X_train, X_valid, y_train_attack, y_valid_attack, y_train_risk, y_valid_risk = train_test_split(
        X_train_val, y_train_val_attack, y_train_val_risk,
        test_size=0.20, # Validation set size (20% of the 80% = 16% of total)
        random_state=42,
        stratify=y_train_val_attack # Stratify again
    )

    # Convert Series to numpy arrays
    y_train_attack = y_train_attack.values
    y_valid_attack = y_valid_attack.values
    y_test_attack = y_test_attack.values
    y_train_risk = y_train_risk.values
    y_valid_risk = y_valid_risk.values
    y_test_risk = y_test_risk.values

    # --- FIX: Fit scaler ONLY on training data ---
    print("Standardizing features (fitting on training data only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid) # Transform validation set
    X_test_scaled = scaler.transform(X_test)   # Transform test set

    print(f"Train shape: {X_train_scaled.shape}, Valid shape: {X_valid_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # Clean up memory
    del train, test, combined_data, data_x, X_train_val, y_train_val_attack, y_train_val_risk
    gc.collect()

    # Train the enhanced neural network - pass validation set
    # Initialize selected_indices before the call
    selected_indices = None
    nn_model, nn_attack_acc, nn_risk_acc, nn_attack_f1, nn_risk_f1, nn_attack_preds, nn_risk_preds, selected_indices = train_enhanced_neural_network(
        X_train_scaled, y_train_attack, y_train_risk,
        X_valid_scaled, y_valid_attack, y_valid_risk, # Pass validation data
        X_test_scaled, y_test_attack, y_test_risk,    # Pass test data
        attack_mapping, risk_mapping, selected_indices # Pass mappings and indices placeholder
    )

    # Generate Confusion Matrices
    print("\nGenerating confusion matrices...")
    attack_class_names = [attack_mapping.get(i, f"Class_{i}") for i in range(len(attack_mapping))]
    risk_class_names = [risk_mapping.get(i, f"Risk_{i}") for i in range(len(risk_mapping))]

    plot_confusion_matrix(
        y_test_attack, nn_attack_preds,
        attack_class_names,
        "Attack Classification (Advanced NN)",
        'W6/attack_nn_advanced_confusion_matrix.png'
    )

    plot_confusion_matrix(
        y_test_risk, nn_risk_preds,
        risk_class_names,
        "Risk Classification (Advanced NN)",
        'W6/risk_nn_advanced_confusion_matrix.png'
    )

    # Final Summary
    print("\n===== FINAL RESULTS SUMMARY (Advanced Model) =====")
    print(f"{'Metric':<30} {'Attack Classification':<25} {'Risk Classification':<25}")
    print("-" * 80)
    print(f"{'Accuracy':<30} {nn_attack_acc:.2f}%{'':<18} {nn_risk_acc:.2f}%")
    print(f"{'Weighted F1 Score':<30} {nn_attack_f1:.4f}{'':<19} {nn_risk_f1:.4f}")
    print("-" * 80)

    print(f"\nTotal script execution time: {(time.time() - overall_start_time)/60:.2f} minutes")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()