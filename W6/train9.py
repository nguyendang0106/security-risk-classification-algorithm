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

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# Enhanced Neural Network with batch normalization and specialized pathways
class EnhancedDualNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes_attack, num_classes_risk, dropout_rate=0.4):
        super(EnhancedDualNN, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn_input = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout = nn.Dropout(dropout_rate)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))

        # Task-specific pathways
        self.attack_hidden1 = nn.Linear(hidden_sizes[-1], 128)
        self.attack_bn1 = nn.BatchNorm1d(128)
        self.attack_hidden2 = nn.Linear(128, 64)
        self.attack_bn2 = nn.BatchNorm1d(64)
        self.attack_head = nn.Linear(64, num_classes_attack)

        self.risk_hidden1 = nn.Linear(hidden_sizes[-1], 64)
        self.risk_bn1 = nn.BatchNorm1d(64)
        self.risk_hidden2 = nn.Linear(64, 32)
        self.risk_bn2 = nn.BatchNorm1d(32)
        self.risk_head = nn.Linear(32, num_classes_risk)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = F.relu(self.bn_input(x))
        x = self.dropout(x)

        # Hidden layers
        for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.bn_layers)):
            x = layer(x)
            x = F.relu(bn(x))
            x = self.dropout(x)

        # Attack classification pathway
        attack = F.relu(self.attack_bn1(self.attack_hidden1(x)))
        attack = self.dropout(attack)
        attack = F.relu(self.attack_bn2(self.attack_hidden2(attack)))
        attack = self.dropout(attack)
        attack_output = self.attack_head(attack)

        # Risk classification pathway
        risk = F.relu(self.risk_bn1(self.risk_hidden1(x)))
        risk = self.dropout(risk)
        risk = F.relu(self.risk_bn2(self.risk_hidden2(risk)))
        risk = self.dropout(risk)
        risk_output = self.risk_head(risk)

        return attack_output, risk_output

# Function to plot confusion matrix with correct labels
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    # Calculate normalized confusion matrix for better visualization
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) # Add epsilon to avoid division by zero

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

    # Add text annotations for counts
    thresh1 = cm.max() / 2. if cm.max() > 0 else 0.5 # Handle case where cm.max() is 0
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

# Function to train the enhanced neural network
def train_enhanced_neural_network(X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk, attack_mapping, risk_mapping): # Added mappings
    """Train an enhanced neural network for dual classification with specialized techniques"""
    print("\n=== Training Enhanced Dual Neural Network ===")
    start_time = time.time()

    # Device configuration - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Improved hyperparameters
    input_size = X_train.shape[1]
    hidden_sizes = [256, 128] # Adjusted hidden layer sizes
    num_classes_attack = len(np.unique(np.concatenate((y_train_attack, y_test_attack)))) # Use combined unique labels
    num_classes_risk = len(np.unique(np.concatenate((y_train_risk, y_test_risk))))     # Use combined unique labels
    num_epochs = 100 # Reduced epochs, rely on early stopping
    batch_size = 256
    learning_rate = 0.0005 # Slightly increased learning rate
    weight_decay = 1e-4

    print(f"Network architecture: input={input_size}, hidden={hidden_sizes}, "
          f"attack_classes={num_classes_attack}, risk_classes={num_classes_risk}")

    # Calculate class weights - smoothed inverse frequency
    print("Computing class weights...")
    # For attack classes
    attack_class_counts = np.bincount(y_train_attack, minlength=num_classes_attack) # Ensure length matches num_classes
    # Smoothed inverse frequency with square root
    attack_weights = np.sqrt(len(y_train_attack) / (attack_class_counts * num_classes_attack + 1e-9)) # Add epsilon
    attack_weights = torch.FloatTensor(attack_weights).to(device)

    # For risk classes
    risk_class_counts = np.bincount(y_train_risk, minlength=num_classes_risk) # Ensure length matches num_classes
    risk_weights = np.sqrt(len(y_train_risk) / (risk_class_counts * num_classes_risk + 1e-9)) # Add epsilon
    risk_weights = torch.FloatTensor(risk_weights).to(device)

    print(f"Attack class weights: {attack_weights.cpu().numpy().round(3)}")
    print(f"Risk class weights: {risk_weights.cpu().numpy().round(3)}")

    # Initialize model with enhanced architecture
    model = EnhancedDualNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes_attack=num_classes_attack,
        num_classes_risk=num_classes_risk,
        dropout_rate=0.4
    ).to(device)

    # Loss and optimizer
    criterion_attack = nn.CrossEntropyLoss(weight=attack_weights)
    criterion_risk = nn.CrossEntropyLoss(weight=risk_weights)

    # Use AdamW optimizer for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True, min_lr=1e-7 # Adjusted patience and min_lr
    )

    # Convert data to PyTorch tensors
    # Ensure data is float32 for features and long for labels
    X_train_tensor = torch.tensor(X_train.values if hasattr(X_train, 'values') else X_train, dtype=torch.float32).to(device)
    y_train_attack_tensor = torch.tensor(y_train_attack.values if hasattr(y_train_attack, 'values') else y_train_attack, dtype=torch.long).to(device)
    y_train_risk_tensor = torch.tensor(y_train_risk.values if hasattr(y_train_risk, 'values') else y_train_risk, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test, dtype=torch.float32).to(device)
    y_test_attack_tensor = torch.tensor(y_test_attack.values if hasattr(y_test_attack, 'values') else y_test_attack, dtype=torch.long).to(device)
    y_test_risk_tensor = torch.tensor(y_test_risk.values if hasattr(y_test_risk, 'values') else y_test_risk, dtype=torch.long).to(device)


    # Create dataset and weighted sampler
    dataset = TensorDataset(X_train_tensor, y_train_attack_tensor, y_train_risk_tensor)

    # Create weighted sampler that considers both tasks
    # Ensure indices are within bounds
    y_train_attack_indices = y_train_attack_tensor.cpu().numpy()
    y_train_risk_indices = y_train_risk_tensor.cpu().numpy()

    # Clamp indices to be within the valid range of weights
    y_train_attack_indices = np.clip(y_train_attack_indices, 0, len(attack_weights) - 1)
    y_train_risk_indices = np.clip(y_train_risk_indices, 0, len(risk_weights) - 1)

    attack_sample_weights = torch.from_numpy(attack_weights[y_train_attack_indices].cpu().numpy())
    risk_sample_weights = torch.from_numpy(risk_weights[y_train_risk_indices].cpu().numpy())

    # Combine weights - prioritize attack classification (70% weight) over risk (30% weight)
    combined_weights = 0.7 * attack_sample_weights + 0.3 * risk_sample_weights
    sampler = WeightedRandomSampler(combined_weights, len(combined_weights), replacement=True) # Ensure replacement=True

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0, # Set to 0 for simplicity, increase if I/O is bottleneck
        pin_memory=False # Set to False if causing issues, True might speed up GPU transfer
    )

    # Training loop with monitoring
    best_f1_attack = 0
    best_f1_risk = 0
    best_combined_f1 = 0
    no_improve_epochs = 0
    patience = 15 # Increased patience

    print(f"Starting training for {num_epochs} epochs (with early stopping)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_attack_weight_avg = 0
        epoch_risk_weight_avg = 0

        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        batch_count = 0
        for batch_x, batch_y_attack, batch_y_risk in progress_bar:
            # Ensure data is on the correct device
            batch_x, batch_y_attack, batch_y_risk = batch_x.to(device), batch_y_attack.to(device), batch_y_risk.to(device)

            # Forward pass
            outputs_attack, outputs_risk = model(batch_x)

            # Calculate losses
            loss_attack = criterion_attack(outputs_attack, batch_y_attack)
            loss_risk = criterion_risk(outputs_risk, batch_y_risk)

            # Adaptive loss weighting - focuses more on the harder task
            loss_ratio = loss_attack.item() / (loss_risk.item() + 1e-8)
            attack_weight = 2.0 / (1.0 + loss_ratio)  # Will be between 0 and 2
            risk_weight = 2.0 - attack_weight         # Complementary weight
            loss = attack_weight * loss_attack + risk_weight * loss_risk

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update loss statistics
            total_loss += loss.item()
            epoch_attack_weight_avg += attack_weight
            epoch_risk_weight_avg += risk_weight
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'a_w': f"{attack_weight:.2f}",
                'r_w': f"{risk_weight:.2f}"
            })

        # Calculate average adaptive weights for the epoch
        avg_attack_weight = epoch_attack_weight_avg / batch_count if batch_count > 0 else 1.0
        avg_risk_weight = epoch_risk_weight_avg / batch_count if batch_count > 0 else 1.0

        # Evaluate model after each epoch
        model.eval()
        with torch.no_grad():
            outputs_attack, outputs_risk = model(X_test_tensor)
            _, attack_preds = torch.max(outputs_attack, 1)
            _, risk_preds = torch.max(outputs_risk, 1)

            # Convert to numpy for metrics calculation
            attack_preds_np = attack_preds.cpu().numpy()
            risk_preds_np = risk_preds.cpu().numpy()
            y_test_attack_np = y_test_attack_tensor.cpu().numpy()
            y_test_risk_np = y_test_risk_tensor.cpu().numpy()

            # Calculate metrics
            attack_acc = accuracy_score(y_test_attack_np, attack_preds_np)
            risk_acc = accuracy_score(y_test_risk_np, risk_preds_np)
            attack_f1 = f1_score(y_test_attack_np, attack_preds_np, average='weighted', zero_division=0)
            risk_f1 = f1_score(y_test_risk_np, risk_preds_np, average='weighted', zero_division=0)

            # Combined F1 score - weighted toward attack classification
            combined_f1 = 0.7 * attack_f1 + 0.3 * risk_f1

            # Calculate validation loss using average weights from training epoch
            val_loss_attack = criterion_attack(outputs_attack, y_test_attack_tensor)
            val_loss_risk = criterion_risk(outputs_risk, y_test_risk_tensor)
            val_loss = avg_attack_weight * val_loss_attack + avg_risk_weight * val_loss_risk

        # Print epoch metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}, "
              f"Attack: {attack_acc*100:.2f}% (F1={attack_f1:.4f}), "
              f"Risk: {risk_acc*100:.2f}% (F1={risk_f1:.4f})")

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Save best model based on combined F1 score
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            best_f1_attack = attack_f1
            best_f1_risk = risk_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'W6/models/best_dual_nn.pt')
            print(f"Model improved (Combined F1: {best_combined_f1:.4f}) - saved checkpoint")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement.")
                break

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    try:
        model.load_state_dict(torch.load('W6/models/best_dual_nn.pt'))
    except FileNotFoundError:
        print("Warning: Best model checkpoint not found. Using the model from the last epoch.")


    # Final evaluation
    print("\n=== Final Model Evaluation (Best Model) ===")
    model.eval()
    with torch.no_grad():
        outputs_attack, outputs_risk = model(X_test_tensor)
        _, attack_preds = torch.max(outputs_attack, 1)
        _, risk_preds = torch.max(outputs_risk, 1)

        # Convert to numpy
        attack_preds_np = attack_preds.cpu().numpy()
        risk_preds_np = risk_preds.cpu().numpy()
        y_test_attack_np = y_test_attack_tensor.cpu().numpy()
        y_test_risk_np = y_test_risk_tensor.cpu().numpy()

        # Calculate final metrics
        final_attack_acc = accuracy_score(y_test_attack_np, attack_preds_np) * 100
        final_risk_acc = accuracy_score(y_test_risk_np, risk_preds_np) * 100
        final_attack_f1 = f1_score(y_test_attack_np, attack_preds_np, average='weighted', zero_division=0)
        final_risk_f1 = f1_score(y_test_risk_np, risk_preds_np, average='weighted', zero_division=0)

        print(f"Attack Classification Accuracy: {final_attack_acc:.2f}%")
        print(f"Attack Classification F1 Score: {final_attack_f1:.4f}")
        print(f"Risk Classification Accuracy: {final_risk_acc:.2f}%")
        print(f"Risk Classification F1 Score: {final_risk_f1:.4f}")

        # Classification reports using the passed mappings
        print("\nAttack Classification Report:")
        # Ensure target names match the number of unique labels predicted/actual
        unique_labels_attack = np.unique(np.concatenate((y_test_attack_np, attack_preds_np)))
        attack_target_names = [attack_mapping.get(i, f"Class_{i}") for i in unique_labels_attack]
        print(classification_report(y_test_attack_np, attack_preds_np, target_names=attack_target_names, zero_division=0))

        print("\nRisk Classification Report:")
        unique_labels_risk = np.unique(np.concatenate((y_test_risk_np, risk_preds_np)))
        risk_target_names = [risk_mapping.get(i, f"Risk_{i}") for i in unique_labels_risk]
        print(classification_report(y_test_risk_np, risk_preds_np, target_names=risk_target_names, zero_division=0))

    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")
    return model, final_attack_acc, final_risk_acc, final_attack_f1, final_risk_f1, attack_preds_np, risk_preds_np


def main():
    print("=== Network Intrusion Detection: Dual Classification NN ===")
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

    # Combine datasets for consistent preprocessing
    combined_data = pd.concat([train, test], ignore_index=True).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")

    # Add risk level based on attack type
    combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP)
    # Handle potential NaN if an attack_cat is not in RISK_MAP (though unlikely with this dataset)
    combined_data['risk_level'].fillna('UNKNOWN', inplace=True)

    # Label encoding
    le_attack = LabelEncoder()
    le_risk = LabelEncoder()

    # Fit and transform labels
    combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
    combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])

    # Create inverse mapping for later use (e.g., confusion matrix labels)
    attack_mapping = {i: cat for i, cat in enumerate(le_attack.classes_)}
    risk_mapping = {i: level for i, level in enumerate(le_risk.classes_)}

    print("\nAttack category mapping:")
    print(attack_mapping)
    print("\nRisk level mapping:")
    print(risk_mapping)

    # Encode categorical features ('proto', 'service', 'state')
    print("\nEncoding categorical features...")
    categorical_features = ['proto', 'service', 'state']
    for feature in categorical_features:
        if feature in combined_data.columns:
            combined_data[feature] = LabelEncoder().fit_transform(combined_data[feature].astype(str)) # Ensure string type
        else:
            print(f"Warning: Feature '{feature}' not found in dataset.")

    # Display class distributions before balancing
    print("\nAttack Type Distribution (Original):")
    print(tabulate(combined_data['attack_cat'].value_counts().reset_index().values.tolist(), headers=['Attack Type', 'Count']))

    print("\nRisk Level Distribution (Original):")
    print(tabulate(combined_data['risk_level'].value_counts().reset_index().values.tolist(), headers=['Risk Level', 'Count']))

    # Feature preparation
    print("\nPreparing features...")
    # Drop original categorical and label columns
    features_to_drop = ['attack_cat', 'risk_level', 'attack_cat_encoded', 'risk_level_encoded']
    data_x = combined_data.drop(columns=features_to_drop, errors='ignore')
    data_y_attack = combined_data['attack_cat_encoded']
    data_y_risk = combined_data['risk_level_encoded']

    # Identify remaining non-numeric columns (if any) - should be none after encoding
    non_numeric_cols = data_x.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"Warning: Non-numeric columns remaining: {list(non_numeric_cols)}. Dropping them.")
        data_x = data_x.drop(columns=non_numeric_cols)

    # Apply standardization
    print("Standardizing features...")
    scaler = StandardScaler()
    # Fit on the entire dataset's features before splitting
    data_x_scaled = pd.DataFrame(
        scaler.fit_transform(data_x),
        columns=data_x.columns,
    )

    # Train-test split with stratification
    # Stratify based on the attack category to maintain distribution
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train_attack, y_test_attack, y_train_risk, y_test_risk = train_test_split(
        data_x_scaled, data_y_attack, data_y_risk,
        test_size=0.20, random_state=42, stratify=data_y_attack
    )

    # Convert Series to numpy arrays for consistency if needed later
    y_train_attack = y_train_attack.values
    y_test_attack = y_test_attack.values
    y_train_risk = y_train_risk.values
    y_test_risk = y_test_risk.values


    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Unique attack classes in train: {len(np.unique(y_train_attack))}")
    print(f"Unique risk classes in train: {len(np.unique(y_train_risk))}")

    # Clean up memory
    del train, test, combined_data, data_x, data_x_scaled
    gc.collect()

    # Train the enhanced neural network
    # Pass the mappings to the training function
    nn_model, nn_attack_acc, nn_risk_acc, nn_attack_f1, nn_risk_f1, nn_attack_preds, nn_risk_preds = train_enhanced_neural_network(
        X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk, attack_mapping, risk_mapping # Pass mappings
    )

    # Generate Confusion Matrices
    print("\nGenerating confusion matrices...")
    # Use the full range of classes defined by the mapping for consistent plotting
    attack_class_names = [attack_mapping.get(i, f"Class_{i}") for i in range(len(attack_mapping))]
    risk_class_names = [risk_mapping.get(i, f"Risk_{i}") for i in range(len(risk_mapping))]

    plot_confusion_matrix(
        y_test_attack, nn_attack_preds,
        attack_class_names,
        "Attack Classification (Neural Network)",
        'W6/attack_nn_confusion_matrix.png'
    )

    plot_confusion_matrix(
        y_test_risk, nn_risk_preds,
        risk_class_names,
        "Risk Classification (Neural Network)",
        'W6/risk_nn_confusion_matrix.png'
    )

    # Final Summary
    print("\n===== FINAL RESULTS SUMMARY =====")
    print(f"{'Metric':<30} {'Attack Classification':<25} {'Risk Classification':<25}")
    print("-" * 80)
    print(f"{'Accuracy':<30} {nn_attack_acc:.2f}%{'':<18} {nn_risk_acc:.2f}%")
    print(f"{'Weighted F1 Score':<30} {nn_attack_f1:.4f}{'':<19} {nn_risk_f1:.4f}")
    print("-" * 80)

    print(f"\nTotal script execution time: {(time.time() - overall_start_time)/60:.2f} minutes")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()