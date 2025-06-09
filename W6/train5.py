import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
from tabulate import tabulate
import warnings
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Scikit-learn imports
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch imports
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

# Define Neural Network Architecture for both attack type and severity classification
class DualOutputNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_attack, num_classes_severity):
        super(DualOutputNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.relu2 = nn.ReLU()
        
        # Separate output heads for attack and severity
        self.fc_attack = nn.Linear(hidden_size//2, num_classes_attack)
        self.fc_severity = nn.Linear(hidden_size//2, num_classes_severity)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        
        attack_out = self.fc_attack(out)
        severity_out = self.fc_severity(out)
        
        return attack_out, severity_out

def main():
    print("Loading datasets...")
    try:
        # Check if the data directory exists
        if not os.path.exists('W6/data'):
            os.makedirs('W6/data', exist_ok=True)
            print("Created directory W6/data")
            print("Please place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in this directory")
            return
            
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files exist in W6/data/")
        return
    

    # Combine datasets
    combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    print("Preview of combined data:")
    print(combined_data.head(5))

    # Add severity_encoded column (missing in original)
    # This is a simple mapping based on attack type - adjust as needed
    severity_map = {
        "Normal": 0, 
        "Analysis": 2,
        "Backdoor": 3,
        "DoS": 3,
        "Exploits": 4,
        "Fuzzers": 1,
        "Generic": 2,
        "Reconnaissance": 1,
        "Shellcode": 4,
        "Worms": 4
    }
    
    # Create severity column based on attack category
    combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)
    print("Added severity_encoded column based on attack types")
    
    # Calculate normal data proportion
    tmp = train.where(train['attack_cat'] == "Normal").dropna()
    print('Train normal proportion:', round(len(tmp['attack_cat'])/len(train['attack_cat']),5))

    tmp = test.where(test['attack_cat'] == "Normal").dropna()
    print('Test normal proportion:', round(len(tmp['attack_cat'])/len(test['attack_cat']),5))

    # Label encoding
    le = LabelEncoder()
    vector = combined_data['attack_cat']
    print("Attack categories:", list(set(list(vector))))

    combined_data['attack_cat'] = le.fit_transform(vector)
    combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
    combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
    combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

    print('\nDescribing attack types:')
    print("Mode:", vector.mode())
    print(f"Mode {np.sum(combined_data['attack_cat'].values==6)/vector.shape[0]:.2f}%")
    print("Looks like 6 is 'normal', but it's not that common")

    # Display counts of each attack type
    counter = collections.Counter(vector)  
    print(tabulate(counter.most_common(), headers=['Type', 'Occurrences']))

    # Create a deep copy
    combined_data = combined_data.copy(deep=True)
    
    # Feature analysis
    lowSTD = list(combined_data.std().to_frame().nsmallest(7, columns=0).index)
    lowCORR = list(combined_data.corr().abs().sort_values('attack_cat')['attack_cat'].nsmallest(7).index)

    exclude = list(set(lowCORR + lowSTD))
    if 'attack_cat' in exclude:
        exclude.remove('attack_cat')
    if 'severity_encoded' in exclude:
        exclude.remove('severity_encoded')

    print('Shape before PCA:', combined_data.shape)
    print('Replace the following with their PCA(3):', exclude)

    # Apply PCA to selected features
    pca = PCA(n_components=3)
    dim_reduct = pca.fit_transform(combined_data[exclude])
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))

    # Remove original features and add PCA results
    combined_data.drop(exclude, axis=1, inplace=True)
    dim_reduction = pd.DataFrame(dim_reduct, columns=['PCA1', 'PCA2', 'PCA3'])
    dim_reduction.index = combined_data.index
    combined_data = combined_data.join(dim_reduction)
    print('Shape after PCA:', combined_data.shape)

    # Feature scaling
    print('combined_data.dur is scaled up by 10,000')
    if 'dur' in combined_data.columns:
        combined_data['dur'] = 10000 * combined_data['dur']
    print(combined_data.head())

    # Prepare data for modeling
    print('Before splitting:', combined_data.shape)
    data_x = combined_data.drop(['attack_cat', 'severity_encoded'], axis=1)  # Fixed: dropping both target columns
    data_y_attack = combined_data['attack_cat']
    data_y_severity = combined_data['severity_encoded']
    print(f"X shape: {data_x.shape}, Y attack shape: {data_y_attack.shape}, Y severity shape: {data_y_severity.shape}")

    # Min-max scaling
    data_x = data_x.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))  # Added epsilon to avoid division by zero

    # Fixed: Proper train_test_split - split twice or all at once
    X_train, X_test, y_train_attack, y_test_attack = train_test_split(
        data_x, data_y_attack, test_size=0.20, random_state=42
    )
    
    # Use same split indices for severity
    _, _, y_train_severity, y_test_severity = train_test_split(
        data_x, data_y_severity, test_size=0.20, random_state=42
    )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Feature selection with RFE for later use
    rfe = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=10).fit(X_train, y_train_attack)
    desiredIndices = np.where(rfe.support_==True)[0]
    selected_features = list(X_train.columns[desiredIndices])
    print("Selected features:", selected_features)

    # Create datasets with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print("Sample X_train:", X_train[:10].values)
    print("Sample X_train_selected:", X_train_selected[:10].values)
    print("Sample y_train_attack:", y_train_attack[:10].values)
    print("Sample y_train_severity:", y_train_severity[:10].values)
    print("Classes attack:", set(y_train_attack))
    print("Classes severity:", set(y_train_severity))

    # Load model and predict
    print("==========  Evaluating Sklearn (.pkl) Models ==========")
    for file in os.listdir("W6/models"):
        if file.endswith(".pkl"):
            try:
                model = joblib.load(f"W6/models/{file}")
                print(f"Evaluating {file}...")
                
                # Determine which data to use based on filename
                if "selected" in file.lower() or "rfe" in file.lower():
                    features_to_use = X_test_selected
                    print("Using selected features")
                else:
                    features_to_use = X_test
                    print("Using all features")
                    
                # Determine which target to evaluate based on filename
                if "severity" in file.lower():
                    # For severity models
                    y_pred = model.predict(features_to_use)
                    acc = accuracy_score(y_test_severity, y_pred)
                    print(f"Severity prediction accuracy: {acc:.4f}")
                    print("Classification report (Severity):")
                    print(classification_report(y_test_severity, y_pred))
                elif "attack" in file.lower() or "ensemble" in file.lower():
                    # For attack models
                    y_pred = model.predict(features_to_use)
                    acc = accuracy_score(y_test_attack, y_pred)
                    print(f"Attack prediction accuracy: {acc:.4f}")
                    print("Classification report (Attack):")
                    print(classification_report(y_test_attack, y_pred))
                else:
                    # Try both if not clear
                    print("Model type not clear from name, trying both targets...")
                    # Try attack target
                    try:
                        y_pred = model.predict(features_to_use)
                        acc_attack = accuracy_score(y_test_attack, y_pred)
                        print(f"Assuming attack target - Accuracy: {acc_attack:.4f}")
                    except:
                        print("Failed to evaluate on attack target")
                    
                    # Try severity target
                    try:
                        y_pred = model.predict(features_to_use)
                        acc_severity = accuracy_score(y_test_severity, y_pred)
                        print(f"Assuming severity target - Accuracy: {acc_severity:.4f}")
                    except:
                        print("Failed to evaluate on severity target")
                
            except Exception as e:
                print(f"Error evaluating {file}: {e}")


    print("\n==========  Evaluating PyTorch (.pth) Models ==========")
    for file in os.listdir("W6/models"):
        if file.endswith(".pth") or file.endswith(".pt"):
            try:
                model_path = f"W6/models/{file}"
                print(f"Evaluating {file}...")
                
                if "selected" in file.lower() or "rfe" in file.lower():
                    features_to_use = X_test_selected
                    print("Using selected features")
                    input_size = len(selected_features)
                else:
                    features_to_use = X_test
                    print("Using all features")
                    input_size = features_to_use.shape[1]
                
                # Determine number of classes
                num_classes_attack = len(np.unique(y_test_attack))
                num_classes_severity = len(np.unique(y_test_severity))
                
                # Create model with matching architecture
                model = DualOutputNN(
                    input_size=input_size,
                    hidden_size=128,
                    num_classes_attack=num_classes_attack,
                    num_classes_severity=num_classes_severity
                )
                
                # Load model weights
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                
                # Convert test data to tensor
                X_test_tensor = torch.FloatTensor(features_to_use.values)
                
                # Make predictions
                with torch.no_grad():
                    attack_outputs, severity_outputs = model(X_test_tensor)
                    _, attack_preds = torch.max(attack_outputs, 1)
                    _, severity_preds = torch.max(severity_outputs, 1)
                
                # Calculate accuracy
                attack_acc = accuracy_score(y_test_attack, attack_preds.numpy())
                severity_acc = accuracy_score(y_test_severity, severity_preds.numpy())
                
                print(f"Attack prediction accuracy: {attack_acc:.4f}")
                print(f"Severity prediction accuracy: {severity_acc:.4f}")
                
                # Show classification reports
                print("Classification report (Attack):")
                print(classification_report(y_test_attack, attack_preds.numpy()))
                print("Classification report (Severity):")
                print(classification_report(y_test_severity, severity_preds.numpy()))
                
                # Display confusion matrix for attack prediction
                print("Confusion Matrix (Attack):")
                cm = confusion_matrix(y_test_attack, attack_preds.numpy())
                print(cm)
                
            except Exception as e:
                print(f"Error evaluating {file}: {e}")

    print("\nModel evaluation complete!")

if __name__ == "__main__":
    main()

# Chưa hoàn thiện