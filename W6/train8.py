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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# For handling imbalanced data
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost and LightGBM for better performance
import xgboost as xgb
import lightgbm as lgb

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

# Function to create more informative features
def engineer_features(df):
    """Engineer additional features to improve model performance"""
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Create log transforms for skewed numerical features
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if data[col].max() > 1000 and data[col].min() >= 0:
            data[f'{col}_log'] = np.log1p(data[col])
    
    # Create ratios between related columns
    if 'sbytes' in data.columns and 'dbytes' in data.columns:
        data['bytes_ratio'] = data['sbytes'] / (data['dbytes'] + 1)
    
    if 'sttl' in data.columns and 'dttl' in data.columns:
        data['ttl_ratio'] = data['sttl'] / (data['dttl'] + 1)
    
    # Time-based features
    if 'dur' in data.columns:
        data['dur_log'] = np.log1p(data['dur'])
        if 'sbytes' in data.columns and 'dbytes' in data.columns:
            data['bytes_per_sec'] = (data['sbytes'] + data['dbytes']) / (data['dur'] + 0.1)
    
    # Flag columns
    for col in data.columns:
        if '_rate' in col or '_cnt' in col:
            data[f'{col}_is_zero'] = (data[col] == 0).astype(int)
    
    return data

# Function to plot feature importance
def plot_feature_importance(features, importances, title='Feature Importance', filename='W6/feature_importance.png'):
    """Plot feature importance from a model"""
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_features = [features[i] for i in indices[:20]]  # Take top 20 features
    top_importances = [importances[i] for i in indices[:20]]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importances, align='center', color='skyblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Feature importance plot saved to '{filename}'")

# Function to plot confusion matrix with correct labels
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate normalized confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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
    thresh1 = cm.max() / 2
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
def train_enhanced_neural_network(X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk):
    """Train an enhanced neural network for dual classification with specialized techniques"""
    print("\n=== Training Enhanced Dual Neural Network ===")
    start_time = time.time()
    
    # Device configuration - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Improved hyperparameters
    input_size = X_train.shape[1]
    hidden_sizes = [256, 128]
    num_classes_attack = len(np.unique(y_train_attack))
    num_classes_risk = len(np.unique(y_train_risk))
    num_epochs = 150
    batch_size = 256
    learning_rate = 0.0003
    weight_decay = 1e-4

    print(f"Network architecture: input={input_size}, hidden={hidden_sizes}, "
          f"attack_classes={num_classes_attack}, risk_classes={num_classes_risk}")

    # Calculate class weights - smoothed inverse frequency
    print("Computing class weights...")
    # For attack classes
    attack_class_counts = np.bincount(y_train_attack)
    # Smoothed inverse frequency with square root 
    attack_weights = np.sqrt(len(y_train_attack) / (attack_class_counts * num_classes_attack))
    attack_weights = torch.FloatTensor(attack_weights).to(device)
    
    # For risk classes
    risk_class_counts = np.bincount(y_train_risk)
    risk_weights = np.sqrt(len(y_train_risk) / (risk_class_counts * num_classes_risk))
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
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(device)
    y_train_attack_tensor = torch.LongTensor(y_train_attack.values if hasattr(y_train_attack, 'values') else y_train_attack).to(device)
    y_train_risk_tensor = torch.LongTensor(y_train_risk.values if hasattr(y_train_risk, 'values') else y_train_risk).to(device)
    
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test).to(device)
    y_test_attack_tensor = torch.LongTensor(y_test_attack.values if hasattr(y_test_attack, 'values') else y_test_attack).to(device)
    y_test_risk_tensor = torch.LongTensor(y_test_risk.values if hasattr(y_test_risk, 'values') else y_test_risk).to(device)
    
    # Create dataset and weighted sampler
    dataset = TensorDataset(X_train_tensor, y_train_attack_tensor, y_train_risk_tensor)
    
    # Create weighted sampler that considers both tasks
    attack_sample_weights = torch.from_numpy(attack_weights[y_train_attack_tensor.cpu()].numpy())
    risk_sample_weights = torch.from_numpy(risk_weights[y_train_risk_tensor.cpu()].numpy())
    # Combine weights - prioritize attack classification (70% weight) over risk (30% weight)
    combined_weights = 0.7 * attack_sample_weights + 0.3 * risk_sample_weights
    sampler = WeightedRandomSampler(combined_weights, len(combined_weights))
    
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    
    # Training loop with monitoring
    best_f1_attack = 0
    best_f1_risk = 0
    best_combined_f1 = 0
    no_improve_epochs = 0
    patience = 15
    
    print(f"Starting training for {num_epochs} epochs (with early stopping)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training loop with progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_x, batch_y_attack, batch_y_risk in progress_bar:
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'a_w': f"{attack_weight:.2f}",
                'r_w': f"{risk_weight:.2f}"
            })
        
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
            attack_f1 = f1_score(y_test_attack_np, attack_preds_np, average='weighted')
            risk_f1 = f1_score(y_test_risk_np, risk_preds_np, average='weighted')
            
            # Combined F1 score - weighted toward attack classification
            combined_f1 = 0.7 * attack_f1 + 0.3 * risk_f1
            
            # Calculate validation loss
            val_loss_attack = criterion_attack(outputs_attack, y_test_attack_tensor)
            val_loss_risk = criterion_risk(outputs_risk, y_test_risk_tensor)
            val_loss = attack_weight * val_loss_attack + risk_weight * val_loss_risk
        
        # Print epoch metrics
        avg_loss = total_loss / len(train_loader)
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
            print(f"Model improved - saved checkpoint")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('W6/models/best_dual_nn.pt'))
    
    # Final evaluation
    print("\n=== Final Model Evaluation ===")
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
        attack_acc = accuracy_score(y_test_attack_np, attack_preds_np) * 100
        risk_acc = accuracy_score(y_test_risk_np, risk_preds_np) * 100
        attack_f1 = f1_score(y_test_attack_np, attack_preds_np, average='weighted')
        risk_f1 = f1_score(y_test_risk_np, risk_preds_np, average='weighted')
        
        print(f"Attack Classification Accuracy: {attack_acc:.2f}%")
        print(f"Attack Classification F1 Score: {attack_f1:.4f}")
        print(f"Risk Classification Accuracy: {risk_acc:.2f}%")
        print(f"Risk Classification F1 Score: {risk_f1:.4f}")
        
        # Classification reports
        print("\nAttack Classification Report:")
        print(classification_report(y_test_attack_np, attack_preds_np))
        
        print("\nRisk Classification Report:")
        print(classification_report(y_test_risk_np, risk_preds_np))
    
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    return model, attack_acc, risk_acc, attack_f1, risk_f1, attack_preds_np, risk_preds_np

# Train optimized models for attack and risk classification
def train_optimized_model(X_train, y_train, X_test, y_test, task="attack", balanced=False):
    """Train an optimized model using the best algorithm for the specific task"""
    print(f"\n=== Training Optimized Model for {task.capitalize()} Classification ===")
    
    if task == "attack":
        # For attack classification: XGBoost with hyper-parameter tuning
        if balanced:
            print("Training on balanced data...")
            # Use class weights instead of SMOTE
            counts = np.bincount(y_train)
            weight_scale = len(y_train) / (len(counts) * counts)
            class_weights = {i: weight_scale[i] for i in range(len(counts))}
            
            # XGBoost optimized for attack classification
            model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Already using class weights
                tree_method='hist',  # Faster algorithm
                n_jobs=2,
                random_state=42
            )
            
            # Train with sample weights
            sample_weights = np.array([class_weights[y] for y in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
        else:
            # Use balanced random forest without SMOTE
            print("Training with BalancedRandomForestClassifier...")
            model = BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced_subsample',
                sampling_strategy='auto',
                replacement=False,
                n_jobs=2,
                random_state=42
            )
            model.fit(X_train, y_train)
    
    else:  # Risk classification
        # For risk classification: LightGBM with optimized parameters
        if balanced:
            print("Training on balanced data...")
            # Use class weights instead of SMOTE
            counts = np.bincount(y_train)
            weight_scale = len(y_train) / (len(counts) * counts)
            class_weights = {i: weight_scale[i] for i in range(len(counts))}
            
            # LightGBM optimized for risk classification
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                boosting_type='dart',  # More robust boosting
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=2,
                importance_type='gain',
                random_state=42
            )
            
            # Train with sample weights
            sample_weights = np.array([class_weights[y] for y in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
        else:
            # Fixed: EasyEnsembleClassifier doesn't accept base_estimator directly
            # Instead, use a different ensemble approach for risk classification
            print("Training with EasyEnsembleClassifier...")
            model = EasyEnsembleClassifier(
                n_estimators=20,
                sampling_strategy='auto',
                replacement=False,
                random_state=42,
                n_jobs=1  # Avoid nested parallelism issues
            )
            model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print(f"\n{task.capitalize()} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy, f1, y_pred

def main():
    print("=== Network Intrusion Detection with Optimized Dual Classification ===")
    start_time = time.time()
    
    # Load data
    print("\nLoading datasets...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files exist in the correct location")
        return

    # Combine datasets for preprocessing
    combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")

    # Add risk level based on attack type
    combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP)
    
    # Label encoding
    le_attack = LabelEncoder()
    le_risk = LabelEncoder()
    
    # Save original categories
    attack_categories = combined_data['attack_cat'].unique()
    risk_levels = combined_data['risk_level'].unique()
    
    combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
    combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])
    
    # Create inverse mapping - for confusion matrix labels
    # Using integers as keys and strings as values
    attack_mapping = {i: cat for i, cat in enumerate(le_attack.classes_)}
    risk_mapping = {i: level for i, level in enumerate(le_risk.classes_)}
    
    print("\nAttack category mapping:")
    for code, cat in attack_mapping.items():
        print(f"  {code}: {cat}")
        
    print("\nRisk level mapping:")
    for code, level in risk_mapping.items():
        print(f"  {code}: {level}")

    # Encode categorical features
    for feature in ['proto', 'service', 'state']:
        if feature in combined_data.columns:
            combined_data[feature] = LabelEncoder().fit_transform(combined_data[feature])

    # Display class distributions
    print("\nAttack Type Distribution:")
    attack_counts = combined_data['attack_cat'].value_counts()
    print(tabulate(attack_counts.reset_index().values.tolist(), headers=['Attack Type', 'Count']))
    
    print("\nRisk Level Distribution:")
    risk_counts = combined_data['risk_level'].value_counts()
    print(tabulate(risk_counts.reset_index().values.tolist(), headers=['Risk Level', 'Count']))

    # Feature preparation
    print("\nPreparing features...")
    data_x = combined_data.drop(['attack_cat', 'risk_level', 'attack_cat_encoded', 'risk_level_encoded'], axis=1)
    data_y_attack = combined_data['attack_cat_encoded']
    data_y_risk = combined_data['risk_level_encoded']
    
    # Feature engineering
    print("Performing feature engineering...")
    data_x_engineered = engineer_features(data_x)
    print(f"Original features: {data_x.shape[1]}, Engineered features: {data_x_engineered.shape[1]}")
    
    # Apply standardization
    print("Standardizing features...")
    scaler = StandardScaler()
    data_x_scaled = pd.DataFrame(
        scaler.fit_transform(data_x_engineered),
        columns=data_x_engineered.columns,
    )

    # Train-test split with stratification
    X_train, X_test, y_train_attack, y_test_attack, y_train_risk, y_test_risk = train_test_split(
        data_x_scaled, data_y_attack, data_y_risk, test_size=0.20, random_state=42, stratify=data_y_attack
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Store accuracies
    accuracies = {}

    # ===== PHASE 1: FEATURE SELECTION WITH BETTER TECHNIQUES =====
    print("\n===== PHASE 1: ADVANCED FEATURE SELECTION =====")
    
    # Use SelectFromModel with XGBoost for feature selection (better than RFE)
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    print("Training XGBoost for feature selection...")
    model.fit(X_train, y_train_attack)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    plot_feature_importance(
        X_train.columns, feature_importances, 
        "Feature Importance for Attack Classification",
        'W6/attack_feature_importance.png'
    )
    
    # Select features using SelectFromModel
    print("Selecting important features...")
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_indices = selector.get_support()
    selected_features = X_train.columns[selected_indices].tolist()
    print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    print("Top 10 selected features:", selected_features[:10])
    
    # ===== PHASE 2: MODEL TRAINING WITHOUT SMOTE =====
    print("\n===== PHASE 2: MODEL TRAINING WITH BUILT-IN CLASS BALANCING =====")
    print("\nTraining models that handle imbalance internally...")
    
    # Train attack classification model using built-in class balancing
    # attack_model, attack_acc, attack_f1, attack_preds = train_optimized_model(
    #     X_train_selected, y_train_attack, X_test_selected, y_test_attack, 
    #     task="attack", balanced=False
    # )
    # accuracies["BalancedModel_Attack"] = attack_acc
    
    # Train risk classification model using built-in class balancing
    # risk_model, risk_acc, risk_f1, risk_preds = train_optimized_model(
    #     X_train_selected, y_train_risk, X_test_selected, y_test_risk, 
    #     task="risk", balanced=False
    # )
    # accuracies["BalancedModel_Risk"] = risk_acc
    
    # ===== PHASE 3: MODEL TRAINING WITH CLASS WEIGHTS =====
    print("\n===== PHASE 3: MODEL TRAINING WITH CLASS WEIGHTS =====")
    print("\nTraining models using sample weights instead of SMOTE...")
    
    # Train attack classification model using class weights
    # attack_model_weighted, attack_acc_weighted, attack_f1_weighted, attack_preds_weighted = train_optimized_model(
    #     X_train_selected, y_train_attack, X_test_selected, y_test_attack,
    #     task="attack", balanced=True
    # )
    # accuracies["WeightedModel_Attack"] = attack_acc_weighted
    
    # Train risk classification model using class weights
    # risk_model_weighted, risk_acc_weighted, risk_f1_weighted, risk_preds_weighted = train_optimized_model(
    #     X_train_selected, y_train_risk, X_test_selected, y_test_risk,
    #     task="risk", balanced=True
    # )
    # accuracies["WeightedModel_Risk"] = risk_acc_weighted
    
    # ===== PHASE 4: NEURAL NETWORK TRAINING =====
    print("\n===== PHASE 4: ENHANCED NEURAL NETWORK TRAINING =====")
    
    # Train enhanced neural network
    nn_model, nn_attack_acc, nn_risk_acc, nn_attack_f1, nn_risk_f1, nn_attack_preds, nn_risk_preds = train_enhanced_neural_network(
        X_train_selected, y_train_attack, y_train_risk, X_test_selected, y_test_attack, y_test_risk
    )
    
    # Store neural network results
    accuracies["NeuralNetwork_Attack"] = nn_attack_acc/100
    accuracies["NeuralNetwork_Risk"] = nn_risk_acc/100
    
    # ===== PHASE 5: GENERATE VISUALIZATIONS =====
    # print("\n===== PHASE 5: CONFUSION MATRIX GENERATION =====")
    
    # # Convert numeric labels to meaningful class names for confusion matrices
    # attack_class_names = [attack_mapping[i] for i in range(len(attack_mapping))]
    # risk_class_names = [risk_mapping[i] for i in range(len(risk_mapping))]
    
    # # Plot confusion matrices for best balanced model
    # print("\nGenerating confusion matrices for balanced models...")
    # plot_confusion_matrix(
    #     y_test_attack, attack_preds,
    #     attack_class_names,
    #     "Attack Classification (Balanced Model)",
    #     'W6/attack_balanced_confusion_matrix.png'
    # )
    
    # plot_confusion_matrix(
    #     y_test_risk, risk_preds,
    #     risk_class_names,
    #     "Risk Classification (Balanced Model)",
    #     'W6/risk_balanced_confusion_matrix.png'
    # )
    
    # # Plot confusion matrices for neural network model
    # print("\nGenerating confusion matrices for neural network...")
    # plot_confusion_matrix(
    #     y_test_attack, nn_attack_preds,
    #     attack_class_names,
    #     "Attack Classification (Neural Network)",
    #     'W6/attack_nn_confusion_matrix.png'
    # )
    
    # plot_confusion_matrix(
    #     y_test_risk, nn_risk_preds,
    #     risk_class_names,
    #     "Risk Classification (Neural Network)",
    #     'W6/risk_nn_confusion_matrix.png'
    # )
    
    # ===== PHASE 6: RESULTS SUMMARY =====
    print("\n===== FINAL RESULTS SUMMARY =====")
    
    # # Add F1 scores to the results
    # model_metrics = {
    #     "BalancedModel_Attack": {"accuracy": attack_acc, "f1": attack_f1},
    #     "WeightedModel_Attack": {"accuracy": attack_acc_weighted, "f1": attack_f1_weighted},
    #     "NeuralNetwork_Attack": {"accuracy": nn_attack_acc/100, "f1": nn_attack_f1},
    #     "BalancedModel_Risk": {"accuracy": risk_acc, "f1": risk_f1},
    #     "WeightedModel_Risk": {"accuracy": risk_acc_weighted, "f1": risk_f1_weighted},
    #     "NeuralNetwork_Risk": {"accuracy": nn_risk_acc/100, "f1": nn_risk_f1}
    # }
    
    # # Separate attack and risk results
    # attack_results = {k: v for k, v in model_metrics.items() if 'Attack' in k}
    # risk_results = {k: v for k, v in model_metrics.items() if 'Risk' in k}
    
    # print("\n--- Attack Classification Results ---")
    # print(tabulate(
    #     [[name, f"{metrics['accuracy']*100:.2f}%", f"{metrics['f1']:.4f}"] 
    #      for name, metrics in sorted(attack_results.items(), key=lambda x: x[1]['f1'], reverse=True)],
    #     headers=["Model", "Accuracy", "F1 Score"]
    # ))
    
    # print("\n--- Risk Classification Results ---")
    # print(tabulate(
    #     [[name, f"{metrics['accuracy']*100:.2f}%", f"{metrics['f1']:.4f}"] 
    #      for name, metrics in sorted(risk_results.items(), key=lambda x: x[1]['f1'], reverse=True)],
    #     headers=["Model", "Accuracy", "F1 Score"]
    # ))
    
    # Save the best performing models
    # import joblib
    # joblib.dump(attack_model, 'W6/models/best_attack_model.pkl')
    # joblib.dump(risk_model, 'W6/models/best_risk_model.pkl')
    # print("\nBest performing models saved to W6/models/")
    
    print(f"\nTotal execution time: {(time.time() - start_time)/60:.2f} minutes")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()