import os
import gc  # Add garbage collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from tabulate import tabulate
import warnings
import joblib
from tqdm import tqdm
import time

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier

# For better models
import xgboost as xgb
import lightgbm as lgb

# For handling imbalanced data
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

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
# Set these environment variables before other imports
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

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

# Neural Network with added feedback and improvements
class ImprovedDualNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes_attack, num_classes_risk, dropout_rate=0.3):
        super(ImprovedDualNN, self).__init__()
        
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
        
        # Task-specific layers
        self.attack_head = nn.Linear(hidden_sizes[-1], num_classes_attack)
        self.risk_head = nn.Linear(hidden_sizes[-1], num_classes_risk)
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = F.relu(self.bn_input(x))
        x = self.dropout(x)
        
        # Hidden layers
        for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.bn_layers)):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output heads
        attack_output = self.attack_head(x)
        risk_output = self.risk_head(x)
        
        return attack_output, risk_output

# Function to apply RFE feature selection
def apply_rfe_feature_selection(X_train, y_train, X_test, n_features=10, verbose=True):
    """Apply RFE feature selection and return reduced datasets"""
    if verbose:
        print(f"\n--- Feature Selection with RFE (top {n_features} features) ---")
    
    # Apply RFE
    start_time = time.time()
    rfe = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=n_features)
    
    # Fit RFE
    with tqdm(total=1, desc="Running RFE", disable=not verbose) as pbar:
        rfe.fit(X_train, y_train)
        pbar.update(1)
    
    # Get selected feature indices
    selected_indices = np.where(rfe.support_)[0]
    
    # Convert to DataFrame if not already
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    
    # Get feature names
    selected_features = list(X_train_df.columns[selected_indices])
    
    # Create reduced datasets
    X_train_rfe = X_train_df.iloc[:, selected_indices]
    X_test_rfe = X_test_df.iloc[:, selected_indices]
    
    if verbose:
        print(f"RFE completed in {time.time() - start_time:.2f} seconds")
        print(f"Selected features: {selected_features}")
        print(f"Reduced shapes: X_train={X_train_rfe.shape}, X_test={X_test_rfe.shape}")
    
    return X_train_rfe, X_test_rfe, selected_features

# Function to train and evaluate models
def train_and_evaluate_models(X_train, y_train, X_test, y_test, prefix="", is_attack=True):
    """Train and evaluate models with specified data"""
    results = {}
    task = "attack" if is_attack else "risk"
    task_name = "Attack" if is_attack else "Risk"
    print(f"\n--- {task_name} Classification Models ({prefix}) ---")
    
    # Define models with restricted parallelism
    if is_attack:
        models = {
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=7,
                class_weight='balanced', n_jobs=2, random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
                scale_pos_weight=5, n_jobs=2, random_state=42, tree_method='hist'
            ),
            "BalancedRF": BalancedRandomForestClassifier(
                n_estimators=100, max_depth=10, max_features='sqrt',
                n_jobs=2, sampling_strategy='auto', bootstrap=True, random_state=42
            )
        }
    else:
        models = {
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=7,
                class_weight='balanced', n_jobs=2, random_state=42
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
                n_jobs=2, random_state=42, tree_method='hist'
            ),
            "BalancedRF": BalancedRandomForestClassifier(
                n_estimators=100, max_depth=10, max_features='sqrt',
                n_jobs=2, sampling_strategy='auto', bootstrap=True, random_state=42
            )
        }

    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name} for {task} classification...")
        start_time_model = time.time()
        
        try:
            # Use tqdm for progress tracking
            with tqdm(total=1) as pbar:
                model.fit(X_train, y_train)
                pbar.update(1)
            
            # Force garbage collection after training
            gc.collect()
            
            # Evaluate
            train_time = time.time() - start_time_model
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Save model performance
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': train_time
            }
            
            # Print results and save model
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Training time: {train_time:.2f} seconds")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            model_filename = f'W6/models/{name}_{task}_{prefix}.pkl'
            joblib.dump(model, model_filename)
            print(f"Model saved to '{model_filename}'")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = {'error': str(e)}
            continue
    
    return results

# Optimized function to create balanced dataset for dual neural network
def create_balanced_dataset_for_nn(X_train, y_train_attack, y_train_risk, random_state=42):
    """Create a balanced dataset for dual neural network training"""
    print("\n=== Creating Balanced Dataset for Dual Neural Network (Optimized) ===")
    start_time = time.time()
    
    # Train a KNN model to predict risk level from features + attack type
    print("Training KNN for risk level prediction...")
    X_combined = X_train.copy()
    X_combined['attack'] = y_train_attack.values
    
    # Create a KNN classifier for risk prediction
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_combined, y_train_risk)
    
    # Apply BorderlineSMOTE for attack balancing
    print("Applying BorderlineSMOTE for balancing attack types...")
    bsmote = BorderlineSMOTE(random_state=random_state, k_neighbors=3)
    
    # Get just the features for SMOTE
    X_features = X_train.copy()
    X_resampled, y_attack_resampled = bsmote.fit_resample(X_features, y_train_attack)
    
    # Use the KNN to predict risk levels for resampled data
    print("Predicting risk levels for balanced dataset...")
    X_with_attack = X_resampled.copy()
    X_with_attack['attack'] = y_attack_resampled
    y_risk_resampled = knn.predict(X_with_attack)
    
    print(f"Created balanced dataset in {time.time() - start_time:.2f} seconds")
    print(f"Dataset shape: {X_resampled.shape}")
    print(f"Attack class distribution: {np.bincount(y_attack_resampled)}")
    print(f"Risk level distribution: {np.bincount(y_risk_resampled)}")
    
    return X_resampled, y_attack_resampled, y_risk_resampled

# Enhanced dual neural network training with regular feedback
def train_dual_neural_network(X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk):
    """Train dual output neural network with enhanced monitoring and feedback"""
    print("\n=== Starting Neural Network Training ===")
    start_time = time.time()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters - optimized for stability
    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]
    num_classes_attack = len(np.unique(y_train_attack))
    num_classes_risk = len(np.unique(y_train_risk))
    num_epochs = 50
    batch_size = 256
    learning_rate = 0.001

    print(f"Network Architecture: input={input_size}, hidden={hidden_sizes}, "
          f"attack_classes={num_classes_attack}, risk_classes={num_classes_risk}")

    # Compute class weights
    print("Computing class weights for imbalanced data...")
    attack_class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_attack),
        y=y_train_attack
    )
    risk_class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_risk),
        y=y_train_risk
    )
    
    attack_weights = torch.FloatTensor(attack_class_weights).to(device)
    risk_weights = torch.FloatTensor(risk_class_weights).to(device)
    
    # Initialize model
    model = ImprovedDualNN(
        input_size=input_size, 
        hidden_sizes=hidden_sizes,
        num_classes_attack=num_classes_attack, 
        num_classes_risk=num_classes_risk
    ).to(device)
    
    # Loss and optimizer
    criterion_attack = nn.CrossEntropyLoss(weight=attack_weights)
    criterion_risk = nn.CrossEntropyLoss(weight=risk_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(device)
    y_train_attack_tensor = torch.LongTensor(y_train_attack.values if hasattr(y_train_attack, 'values') else y_train_attack).to(device)
    y_train_risk_tensor = torch.LongTensor(y_train_risk.values if hasattr(y_train_risk, 'values') else y_train_risk).to(device)
    
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test).to(device)
    y_test_attack_tensor = torch.LongTensor(y_test_attack.values if hasattr(y_test_attack, 'values') else y_test_attack).to(device)
    y_test_risk_tensor = torch.LongTensor(y_test_risk.values if hasattr(y_test_risk, 'values') else y_test_risk).to(device)
    
    # Create data loader - FIX: Set num_workers=0 to avoid multiprocessing issues
    dataset = TensorDataset(X_train_tensor, y_train_attack_tensor, y_train_risk_tensor)
    attack_class_counts = np.bincount(y_train_attack_tensor.cpu().numpy())
    attack_weights_per_sample = 1. / attack_class_counts[y_train_attack_tensor.cpu().numpy()]
    sampler = WeightedRandomSampler(attack_weights_per_sample, len(attack_weights_per_sample))
    
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # No multiprocessing for DataLoader
        pin_memory=False  # Disable pinned memory
    )
    
    # Training loop
    best_loss = float('inf')
    no_improve_epochs = 0
    patience = 7
    
    try:
        print(f"Starting training for {num_epochs} epochs (with early stopping)...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            # Progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_x, batch_y_attack, batch_y_risk in progress_bar:
                # Forward pass
                outputs_attack, outputs_risk = model(batch_x)
                loss_attack = criterion_attack(outputs_attack, batch_y_attack)
                loss_risk = criterion_risk(outputs_risk, batch_y_risk)
                loss = loss_attack + loss_risk
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                outputs_attack, outputs_risk = model(X_test_tensor)
                _, attack_preds = torch.max(outputs_attack, 1)
                _, risk_preds = torch.max(outputs_risk, 1)
                
                attack_acc = (attack_preds == y_test_attack_tensor).float().mean().item() * 100
                risk_acc = (risk_preds == y_test_risk_tensor).float().mean().item() * 100
                
                val_loss_attack = criterion_attack(outputs_attack, y_test_attack_tensor)
                val_loss_risk = criterion_risk(outputs_risk, y_test_risk_tensor)
                val_loss = val_loss_attack + val_loss_risk
            
            # Print epoch summary
            epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Attack Acc: {attack_acc:.2f}%, "
                  f"Risk Acc: {risk_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve_epochs = 0
                torch.save(model.state_dict(), 'W6/models/best_dual_model.pt')
                print(f"Model improved - saved checkpoint")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Force garbage collection after each epoch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Load the best model for final evaluation
        model.load_state_dict(torch.load('W6/models/best_dual_model.pt'))
        
        # Final evaluation
        print("\n=== Final Model Evaluation ===")
        model.eval()
        with torch.no_grad():
            outputs_attack, outputs_risk = model(X_test_tensor)
            _, attack_preds = torch.max(outputs_attack, 1)
            _, risk_preds = torch.max(outputs_risk, 1)
            
            attack_acc = (attack_preds == y_test_attack_tensor).float().mean().item() * 100
            risk_acc = (risk_preds == y_test_risk_tensor).float().mean().item() * 100
            
            print(f"Attack Classification Accuracy: {attack_acc:.2f}%")
            print(f"Risk Classification Accuracy: {risk_acc:.2f}%")
            
            # Generate classification reports
            attack_preds_np = attack_preds.cpu().numpy()
            risk_preds_np = risk_preds.cpu().numpy()
            y_test_attack_np = y_test_attack_tensor.cpu().numpy()
            y_test_risk_np = y_test_risk_tensor.cpu().numpy()
            
            print("\nAttack Classification Report:")
            print(classification_report(y_test_attack_np, attack_preds_np))
            
            print("\nRisk Classification Report:")
            print(classification_report(y_test_risk_np, risk_preds_np))
        
    except Exception as e:
        print(f"Error during neural network training: {str(e)}")
    
    # Clean up resources
    del X_train_tensor, y_train_attack_tensor, y_train_risk_tensor
    del X_test_tensor, y_test_attack_tensor, y_test_risk_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    return model, attack_acc, risk_acc
def main():
    print("=== Network Intrusion Detection with Optimized Performance ===")
    start_time = time.time()
    
    try: 
        # Load data
        print("\nLoading datasets...")
        try:
            train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
            test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return

        # Combine datasets for preprocessing
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
        print(f"Combined data shape: {combined_data.shape}")

        # Add risk level
        combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP)
        
        # Label encoding
        le_attack = LabelEncoder()
        le_risk = LabelEncoder()
        
        # Save original categories
        attack_categories = combined_data['attack_cat'].unique()
        risk_levels = combined_data['risk_level'].unique()
        
        combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
        combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])
        
        # Create label maps
        attack_mapping = dict(zip(le_attack.classes_, range(len(le_attack.classes_))))
        risk_mapping = dict(zip(le_risk.classes_, range(len(le_risk.classes_))))
        
        print("\nAttack category mapping:")
        for cat, code in attack_mapping.items():
            print(f"  {code}: {cat}")
            
        print("\nRisk level mapping:")
        for level, code in risk_mapping.items():
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
        
        # Apply preprocessing (standardization works better than min-max scaling for most models)
        scaler = StandardScaler()
        data_x_scaled = pd.DataFrame(
            scaler.fit_transform(data_x),
            columns=data_x.columns,
        )

        # Train-test split with stratification
        X_train, X_test, y_train_attack, y_test_attack, y_train_risk, y_test_risk = train_test_split(
            data_x_scaled, data_y_attack, data_y_risk, test_size=0.20, random_state=42, stratify=data_y_attack
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Store results for comparison
        all_results = {
            'original': {
                'attack': {},
                'risk': {}
            },
            'rfe': {
                'attack': {},
                'risk': {}
            },
            'neural_network': {
                'original': {},
                'rfe': {}
            }
        }
        
        # Choose which parts of the pipeline to run
        run_original_models = False  # Set to True to run models with all features
        run_rfe_models = True       # Set to True to run models with RFE features
        run_nn_original = True      # Set to True to run neural network with all features
        run_nn_rfe = True           # Set to True to run neural network with RFE features
        
        # ===== PART 1: TRAINING WITH ALL FEATURES =====
        print("\n===== PART 1: TRAINING WITH ALL FEATURES =====")
        
        # Balance original training data for attack classification
        print("\nBalancing data for attack classification...")
        bsmote = BorderlineSMOTE(random_state=42, k_neighbors=5)
        X_train_attack_resampled, y_train_attack_resampled = bsmote.fit_resample(X_train, y_train_attack)
        
        # Balance original training data for risk classification
        print("\nBalancing data for risk classification...")
        smote = SMOTE(random_state=42)
        X_train_risk_resampled, y_train_risk_resampled = smote.fit_resample(X_train, y_train_risk)
        
        # Display balancing results
        print(f"\nOriginal attack distribution: {np.bincount(y_train_attack)}")
        print(f"Resampled attack distribution: {np.bincount(y_train_attack_resampled)}")
        print(f"\nOriginal risk distribution: {np.bincount(y_train_risk)}")
        print(f"Resampled risk distribution: {np.bincount(y_train_risk_resampled)}")
        
        # Train models for attack classification with all features
        if run_original_models:
            # all_results['original']['attack'] = train_and_evaluate_models(
            #     X_train_attack_resampled, y_train_attack_resampled, 
            #     X_test, y_test_attack,
            #     prefix="original", is_attack=True
            # )
            
            # Clean up to free memory
            del X_train_attack_resampled, y_train_attack_resampled
            gc.collect()
            
            # Train models for risk classification with all features
            # all_results['original']['risk'] = train_and_evaluate_models(
            #     X_train_risk_resampled, y_train_risk_resampled,
            #     X_test, y_test_risk,
            #     prefix="original", is_attack=False
            # )
            
            # Clean up to free memory
            del X_train_risk_resampled, y_train_risk_resampled
            gc.collect()
        
        # ===== PART 2: TRAINING WITH RFE SELECTED FEATURES =====
        print("\n===== PART 2: TRAINING WITH RFE SELECTED FEATURES =====")
        
        # Apply RFE feature selection
        X_train_rfe, X_test_rfe, selected_features = apply_rfe_feature_selection(
            X_train, y_train_attack, X_test, n_features=10
        )
        
        if run_rfe_models:
            # Balance RFE training data for attack classification
            print("\nBalancing RFE data for attack classification...")
            X_train_attack_rfe_resampled, y_train_attack_rfe_resampled = bsmote.fit_resample(
                X_train_rfe, y_train_attack
            )
            
            # Train models for attack classification with RFE features
            # all_results['rfe']['attack'] = train_and_evaluate_models(
            #     X_train_attack_rfe_resampled, y_train_attack_rfe_resampled,
            #     X_test_rfe, y_test_attack,
            #     prefix="rfe", is_attack=True
            # )
            
            # Clean up to free memory
            del X_train_attack_rfe_resampled, y_train_attack_rfe_resampled
            gc.collect()
            
            # Balance RFE training data for risk classification
            print("\nBalancing RFE data for risk classification...")
            X_train_risk_rfe_resampled, y_train_risk_rfe_resampled = smote.fit_resample(
                X_train_rfe, y_train_risk
            )
            
            # Train models for risk classification with RFE features
            # all_results['rfe']['risk'] = train_and_evaluate_models(
            #     X_train_risk_rfe_resampled, y_train_risk_rfe_resampled,
            #     X_test_rfe, y_test_risk,
            #     prefix="rfe", is_attack=False
            # )
            
            # Clean up to free memory
            del X_train_risk_rfe_resampled, y_train_risk_rfe_resampled
            gc.collect()
        
        # ===== PART 3: NEURAL NETWORK TRAINING =====
        print("\n===== PART 3: NEURAL NETWORK TRAINING =====")
        
        if run_nn_original:
            # NEURAL NETWORK WITH ALL FEATURES
            print("\n--- Neural Network with All Features ---")
            
            # Create balanced dataset for dual neural network (optimized method)
            X_nn_balanced, y_attack_nn_balanced, y_risk_nn_balanced = create_balanced_dataset_for_nn(
                X_train, y_train_attack, y_train_risk
            )
            
            # Train neural network with all features
            nn_model_full, nn_attack_acc_full, nn_risk_acc_full = train_dual_neural_network(
                X_nn_balanced, y_attack_nn_balanced, y_risk_nn_balanced,
                X_test, y_test_attack, y_test_risk
            )
            
            all_results['neural_network']['original'] = {
                'attack_accuracy': nn_attack_acc_full/100,
                'risk_accuracy': nn_risk_acc_full/100
            }
            
            # Clean up to free memory
            del X_nn_balanced, y_attack_nn_balanced, y_risk_nn_balanced, nn_model_full
            gc.collect()
        
        if run_nn_rfe:
            # NEURAL NETWORK WITH RFE FEATURES
            print("\n--- Neural Network with RFE Features ---")
            
            # Create balanced dataset for dual neural network with RFE features
            X_nn_rfe_balanced, y_attack_nn_rfe_balanced, y_risk_nn_rfe_balanced = create_balanced_dataset_for_nn(
                X_train_rfe, y_train_attack, y_train_risk
            )
            
            # Train neural network with RFE features
            nn_model_rfe, nn_attack_acc_rfe, nn_risk_acc_rfe = train_dual_neural_network(
                X_nn_rfe_balanced, y_attack_nn_rfe_balanced, y_risk_nn_rfe_balanced,
                X_test_rfe, y_test_attack, y_test_risk
            )
            
            all_results['neural_network']['rfe'] = {
                'attack_accuracy': nn_attack_acc_rfe/100,
                'risk_accuracy': nn_risk_acc_rfe/100
            }
            
            # Clean up to free memory
            del X_nn_rfe_balanced, y_attack_nn_rfe_balanced, y_risk_nn_rfe_balanced, nn_model_rfe
            gc.collect()
        
        # ===== RESULTS COMPARISON =====
        print("\n===== RESULTS COMPARISON =====")
        
        # Compare attack classification results
        print("\nAttack Classification Results:")
        attack_results = []
        
        # Original feature results
        if run_original_models:
            for name, result in sorted(all_results['original']['attack'].items(), 
                                    key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in result and 'f1_score' in result:
                    attack_results.append([
                        f"{name} (Original)", 
                        f"{result['accuracy']:.4f}", 
                        f"{result['f1_score']:.4f}"
                    ])
        
        # RFE feature results
        if run_rfe_models:
            for name, result in sorted(all_results['rfe']['attack'].items(), 
                                    key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in result and 'f1_score' in result:
                    attack_results.append([
                        f"{name} (RFE)", 
                        f"{result['accuracy']:.4f}", 
                        f"{result['f1_score']:.4f}"
                    ])
        
        # Neural network results
        if run_nn_original and 'attack_accuracy' in all_results['neural_network']['original']:
            attack_results.append([
                "Neural Network (Original)",
                f"{all_results['neural_network']['original']['attack_accuracy']:.4f}",
                "N/A"
            ])
        if run_nn_rfe and 'attack_accuracy' in all_results['neural_network']['rfe']:
            attack_results.append([
                "Neural Network (RFE)",
                f"{all_results['neural_network']['rfe']['attack_accuracy']:.4f}",
                "N/A"
            ])
        
        print(tabulate(attack_results, headers=["Model", "Accuracy", "F1 Score"]))
        
        # Compare risk classification results
        print("\nRisk Classification Results:")
        risk_results = []
        
        # Original feature results
        if run_original_models:
            for name, result in sorted(all_results['original']['risk'].items(), 
                                    key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in result and 'f1_score' in result:
                    risk_results.append([
                        f"{name} (Original)", 
                        f"{result['accuracy']:.4f}", 
                        f"{result['f1_score']:.4f}"
                    ])
        
        # RFE feature results
        if run_rfe_models:
            for name, result in sorted(all_results['rfe']['risk'].items(), 
                                    key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in result and 'f1_score' in result:
                    risk_results.append([
                        f"{name} (RFE)", 
                        f"{result['accuracy']:.4f}", 
                        f"{result['f1_score']:.4f}"
                    ])
        
        # Neural network results
        if run_nn_original and 'risk_accuracy' in all_results['neural_network']['original']:
            risk_results.append([
                "Neural Network (Original)",
                f"{all_results['neural_network']['original']['risk_accuracy']:.4f}",
                "N/A"
            ])
        if run_nn_rfe and 'risk_accuracy' in all_results['neural_network']['rfe']:
            risk_results.append([
                "Neural Network (RFE)",
                f"{all_results['neural_network']['rfe']['risk_accuracy']:.4f}",
                "N/A"
            ])
        
        print(tabulate(risk_results, headers=["Model", "Accuracy", "F1 Score"]))
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure proper cleanup
        gc.collect()
        print(f"\nTotal execution time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()