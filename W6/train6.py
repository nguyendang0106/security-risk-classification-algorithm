import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from tabulate import tabulate
import warnings
import joblib

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

# PyTorch imports
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure models directory exists
os.makedirs('W6/models', exist_ok=True)

# Risk level mapping from attack categories
RISK_MAP = {
    "Normal": "UNKNOWN", 
    "Analysis": "MEDIUM",
    "Backdoor": "HIGH",
    "DoS": "HIGH",
    "Exploits": "CRITICAL",
    "Fuzzers": "LOW",
    "Generic": "MEDIUM",
    "Reconnaissance":"LOW",
    "Shellcode": "CRITICAL",
    "Worms": "CRITICAL"
}

# Dual-output neural network for attack type and risk level classification
class DualOutputNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk):
        super(DualOutputNN, self).__init__()
        # Shared layers
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Separate output heads
        self.attack_head = nn.Linear(hidden_size_2, num_classes_attack)
        self.risk_head = nn.Linear(hidden_size_2, num_classes_risk)
    
    def forward(self, x):
        # Shared layers
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        
        # Separate outputs
        attack_out = self.attack_head(out)
        risk_out = self.risk_head(out)
        
        return attack_out, risk_out

# Function to plot feature importance
def plot_feature_importance(features, importances, task="Attack Type"):
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title(f"Feature Importance for {task} Classification")
    plt.gca().invert_yaxis()
    plt.savefig(f'W6/feature_importance_{task.lower().replace(" ", "_")}.png')
    print(f"Feature importance plot saved to 'W6/feature_importance_{task.lower().replace(' ', '_')}.png'")
    plt.close()

# Training function for dual neural network
def train_dual_neural_network(X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 128
    hidden_size_2 = 64
    num_classes_attack = len(set(y_train_attack))
    num_classes_risk = len(set(y_train_risk))
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001

    print(f"Neural Network parameters: input_size={input_size}, hidden_size={hidden_size}, "
          f"hidden_size_2={hidden_size_2}, attack_classes={num_classes_attack}, risk_classes={num_classes_risk}")

    # Initialize model
    model = DualOutputNN(input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk).to(device)

    # Loss and optimizer
    criterion_attack = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to numpy arrays
    X_train_vals = X_train.values.astype(np.float32)
    y_train_attack_vals = y_train_attack.values.astype(np.int64)
    y_train_risk_vals = y_train_risk.values.astype(np.int64)
    X_test_vals = X_test.values.astype(np.float32)
    y_test_attack_vals = y_test_attack.values.astype(np.int64)
    y_test_risk_vals = y_test_risk.values.astype(np.int64)

    # Train the model
    n_total_steps = len(X_train_vals)
    print(f"Starting neural network training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_attack = 0
        epoch_loss_risk = 0
        total_loss = 0
        
        # Process in batches to avoid memory issues
        for i in range(0, X_train_vals.shape[0], batch_size):
            batch_end = min(i + batch_size, X_train_vals.shape[0])
            x = torch.FloatTensor(X_train_vals[i:batch_end]).to(device)
            y_attack = torch.LongTensor(y_train_attack_vals[i:batch_end]).to(device)
            y_risk = torch.LongTensor(y_train_risk_vals[i:batch_end]).to(device)
            
            # Forward pass
            attack_outputs, risk_outputs = model(x)
            loss_attack = criterion_attack(attack_outputs, y_attack)
            loss_risk = criterion_risk(risk_outputs, y_risk)
            
            # Combined loss
            loss = loss_attack + loss_risk
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss_attack += loss_attack.item() * (batch_end - i)
            epoch_loss_risk += loss_risk.item() * (batch_end - i)
            total_loss += loss.item() * (batch_end - i)
        
        # Print epoch stats
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Attack Loss: {epoch_loss_attack/n_total_steps:.7f}, '
                  f'Risk Loss: {epoch_loss_risk/n_total_steps:.7f}, '
                  f'Total Loss: {total_loss/n_total_steps:.7f}')

    # Test the model
    print("Evaluating neural network...")
    model.eval()
    with torch.no_grad():
        n_correct_attack = 0
        n_correct_risk = 0
        n_samples = 0
        
        for i in range(0, X_test_vals.shape[0], batch_size):
            batch_end = min(i + batch_size, X_test_vals.shape[0])
            x = torch.FloatTensor(X_test_vals[i:batch_end]).to(device)
            y_attack = torch.LongTensor(y_test_attack_vals[i:batch_end]).to(device)
            y_risk = torch.LongTensor(y_test_risk_vals[i:batch_end]).to(device)
            
            attack_outputs, risk_outputs = model(x)
            
            _, attack_predicted = torch.max(attack_outputs, 1)
            _, risk_predicted = torch.max(risk_outputs, 1)
            
            n_samples += y_attack.size(0)
            n_correct_attack += (attack_predicted == y_attack).sum().item()
            n_correct_risk += (risk_predicted == y_risk).sum().item()
        
        acc_attack = 100.0 * n_correct_attack / n_samples if n_samples > 0 else 0
        acc_risk = 100.0 * n_correct_risk / n_samples if n_samples > 0 else 0
        
        print(f'Attack Classification Accuracy: {acc_attack:.7f}%')
        print(f'Risk Level Classification Accuracy: {acc_risk:.7f}%')
    
    # Save the model
    # torch.save(model.state_dict(), 'W6/models/dual_neural_network.pt')
    # print("Neural network model saved to 'W6/models/dual_neural_network.pt'")
    
    return model, acc_attack, acc_risk

def main():
    print("Loading datasets...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files exist in the correct location")
        return

    # Combine datasets
    combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")

    # Add risk level based on attack category
    combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP)
    print("Risk level distribution:")
    print(combined_data['risk_level'].value_counts())

    # Display attack category distribution
    print("Attack category distribution:")
    print(combined_data['attack_cat'].value_counts())

    # Calculate normal data proportion
    normal_train = train[train['attack_cat'] == "Normal"]
    normal_test = test[test['attack_cat'] == "Normal"]
    print('Train normal proportion:', round(len(normal_train) / len(train), 5))
    print('Test normal proportion:', round(len(normal_test) / len(test), 5))

    # Label encoding
    le_attack = LabelEncoder()
    le_risk = LabelEncoder()
    
    # Original categories for reference
    attack_categories = combined_data['attack_cat'].unique()
    risk_levels = combined_data['risk_level'].unique()
    print("Attack categories:", attack_categories)
    print("Risk levels:", risk_levels)
    
    # Encode targets
    combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
    combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])
    
    # Encode categorical features
    combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
    combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
    combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

    # Display attack type and risk level mapping
    attack_mapping = dict(zip(le_attack.classes_, le_attack.transform(le_attack.classes_)))
    risk_mapping = dict(zip(le_risk.classes_, le_risk.transform(le_risk.classes_)))
    print("Attack category mapping:", attack_mapping)
    print("Risk level mapping:", risk_mapping)

    # Display counts of each attack type
    attack_counts = collections.Counter(combined_data['attack_cat'])
    risk_counts = collections.Counter(combined_data['risk_level'])
    print("\nAttack Type Distribution:")
    print(tabulate(attack_counts.most_common(), headers=['Attack Type', 'Count']))
    print("\nRisk Level Distribution:")
    print(tabulate(risk_counts.most_common(), headers=['Risk Level', 'Count']))

    # Feature analysis
    # Get only numeric columns for standard deviation calculation
    numeric_cols = combined_data.select_dtypes(include=np.number).columns
    lowSTD = list(combined_data[numeric_cols].std().to_frame().nsmallest(7, columns=0).index)
    # Get only numeric columns for correlation calculation
    if 'attack_cat_encoded' in numeric_cols:
        lowCORR = list(combined_data[numeric_cols].corr().abs().sort_values('attack_cat_encoded')['attack_cat_encoded'].nsmallest(7).index)
    else:
        print("Warning: attack_cat_encoded not found in numeric columns. Using empty list for lowCORR.")
        lowCORR = []

    exclude = list(set(lowCORR + lowSTD))
    for col in ['attack_cat_encoded', 'risk_level_encoded', 'attack_cat', 'risk_level']:
        if col in exclude:
            exclude.remove(col)

    print('Shape before PCA:', combined_data.shape)
    print('Replace the following with their PCA(3):', exclude)

    # Apply PCA to selected features
    pca = PCA(n_components=3)
    dim_reduct = pca.fit_transform(combined_data[exclude])
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))

    # Remove original features and add PCA results
    combined_data.drop(exclude, axis=1, inplace=True)
    dim_reduction = pd.DataFrame(dim_reduct, columns=['PCA1', 'PCA2', 'PCA3'], index=combined_data.index)
    combined_data = pd.concat([combined_data, dim_reduction], axis=1)
    print('Shape after PCA:', combined_data.shape)

    # Feature scaling
    if 'dur' in combined_data.columns:
        combined_data['dur'] = 10000 * combined_data['dur']

    # Prepare data for modeling
    data_x = combined_data.drop(['attack_cat', 'risk_level', 'attack_cat_encoded', 'risk_level_encoded'], axis=1)
    data_y_attack = combined_data['attack_cat_encoded']
    data_y_risk = combined_data['risk_level_encoded']
    print(f"X shape: {data_x.shape}, Attack Y shape: {data_y_attack.shape}, Risk Y shape: {data_y_risk.shape}")

    # Min-max scaling
    data_x = data_x.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

    # Train-test split - stratify on attack type to maintain distribution
    X_train, X_test, y_train_attack, y_test_attack, y_train_risk, y_test_risk = train_test_split(
        data_x, data_y_attack, data_y_risk, test_size=0.20, random_state=42, stratify=data_y_attack
    )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Store accuracies
    attack_accuracies = {}
    risk_accuracies = {}

    # ========== Initial model training for attack type classification ==========
    print("\n--- Initial Model Training (Attack Type) ---")
    models_attack = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False, random_state=42)
    }

    for name, clf in models_attack.items():
        clf.fit(X_train, y_train_attack)
        acc = clf.score(X_test, y_test_attack)
        attack_accuracies[name] = acc
        print(f"Attack accuracy: {acc:.7f} [{name}]")
        # joblib.dump(clf, f'W6/models/{name}_attack.pkl')

    # Train ensemble classifier for attack
    attack_estimators = [(name, clf) for name, clf in models_attack.items()]
    eclf_attack = VotingClassifier(estimators=attack_estimators, voting='hard')
    eclf_attack.fit(X_train, y_train_attack)
    acc = eclf_attack.score(X_test, y_test_attack)
    attack_accuracies["Ensemble"] = acc
    print(f"Attack accuracy: {acc:.7f} [Ensemble]")
    # joblib.dump(eclf_attack, 'W6/models/Ensemble_attack.pkl')

    # ========== Initial model training for risk level classification ==========
    print("\n--- Initial Model Training (Risk Level) ---")
    models_risk = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False, random_state=42)
    }

    for name, clf in models_risk.items():
        clf.fit(X_train, y_train_risk)
        acc = clf.score(X_test, y_test_risk)
        risk_accuracies[name] = acc
        print(f"Risk accuracy: {acc:.7f} [{name}]")
        # joblib.dump(clf, f'W6/models/{name}_risk.pkl')

    # Train ensemble classifier for risk
    risk_estimators = [(name, clf) for name, clf in models_risk.items()]
    eclf_risk = VotingClassifier(estimators=risk_estimators, voting='hard')
    eclf_risk.fit(X_train, y_train_risk)
    acc = eclf_risk.score(X_test, y_test_risk)
    risk_accuracies["Ensemble"] = acc
    print(f"Risk accuracy: {acc:.7f} [Ensemble]")
    # joblib.dump(eclf_risk, 'W6/models/Ensemble_risk.pkl')

    # ========== Feature selection with RFE ==========
    print("\n--- Feature Selection with RFE ---")
    rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10).fit(X_train, y_train_attack)
    desiredIndices = np.where(rfe.support_==True)[0]
    selected_features = list(X_train.columns[desiredIndices])
    print("Selected features:", selected_features)

    X_train_RFE = X_train[selected_features]
    X_test_RFE = X_test[selected_features]
    
    print(f"RFE selected {len(selected_features)} features")
    print(f"X_train_RFE shape: {X_train_RFE.shape}, X_test_RFE shape: {X_test_RFE.shape}")

    # ========== Training with selected features for attack type ==========
    print("\n--- Training with Selected Features (Attack Type) ---")
    models_attack_rfe = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False, random_state=42)
    }

    for name, clf in models_attack_rfe.items():
        clf.fit(X_train_RFE, y_train_attack)
        acc = clf.score(X_test_RFE, y_test_attack)
        attack_accuracies[f"{name}_RFE"] = acc
        print(f"Attack accuracy with RFE: {acc:.7f} [{name}]")
        # joblib.dump(clf, f'W6/models/{name}_RFE_attack.pkl')

    # Train ensemble classifier for attack with RFE
    attack_rfe_estimators = [(name, clf) for name, clf in models_attack_rfe.items()]
    eclf_attack_rfe = VotingClassifier(estimators=attack_rfe_estimators, voting='hard')
    eclf_attack_rfe.fit(X_train_RFE, y_train_attack)
    acc = eclf_attack_rfe.score(X_test_RFE, y_test_attack)
    attack_accuracies["Ensemble_RFE"] = acc
    print(f"Attack accuracy with RFE: {acc:.7f} [Ensemble]")
    # joblib.dump(eclf_attack_rfe, 'W6/models/Ensemble_RFE_attack.pkl')

    # ========== Training with selected features for risk level ==========
    print("\n--- Training with Selected Features (Risk Level) ---")
    models_risk_rfe = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False, random_state=42)
    }

    for name, clf in models_risk_rfe.items():
        clf.fit(X_train_RFE, y_train_risk)
        acc = clf.score(X_test_RFE, y_test_risk)
        risk_accuracies[f"{name}_RFE"] = acc
        print(f"Risk accuracy with RFE: {acc:.7f} [{name}]")
        # joblib.dump(clf, f'W6/models/{name}_RFE_risk.pkl')

    # Train ensemble classifier for risk with RFE
    risk_rfe_estimators = [(name, clf) for name, clf in models_risk_rfe.items()]
    eclf_risk_rfe = VotingClassifier(estimators=risk_rfe_estimators, voting='hard')
    eclf_risk_rfe.fit(X_train_RFE, y_train_risk)
    acc = eclf_risk_rfe.score(X_test_RFE, y_test_risk)
    risk_accuracies["Ensemble_RFE"] = acc
    print(f"Risk accuracy with RFE: {acc:.7f} [Ensemble]")
    # joblib.dump(eclf_risk_rfe, 'W6/models/Ensemble_RFE_risk.pkl')

    # ========== Generate feature importance plots ==========
    # Feature importance for attack classification
    RFC_RFE = RandomForestClassifier(n_estimators=50, random_state=42)
    RFC_RFE.fit(X_train_RFE, y_train_attack)
    attack_importances = RFC_RFE.feature_importances_
    plot_feature_importance(selected_features, attack_importances, "Attack Type")

    # Feature importance for risk classification
    RFC_RFE.fit(X_train_RFE, y_train_risk)
    risk_importances = RFC_RFE.feature_importances_
    plot_feature_importance(selected_features, risk_importances, "Risk Level")

    # ========== Train dual neural network model ==========
    print("\n--- Training Dual-Output Neural Network ---")
    nn_model, nn_acc_attack, nn_acc_risk = train_dual_neural_network(
        X_train_RFE, y_train_attack, y_train_risk, 
        X_test_RFE, y_test_attack, y_test_risk
    )
    
    attack_accuracies["Neural Network"] = nn_acc_attack/100
    risk_accuracies["Neural Network"] = nn_acc_risk/100

    # ========== Print summary of all models ==========
    print("\n--- Attack Classification Accuracy Summary ---")
    for model_name, accuracy in sorted(attack_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {accuracy:.7f}")
    
    print("\n--- Risk Level Classification Accuracy Summary ---")
    for model_name, accuracy in sorted(risk_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {accuracy:.7f}")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()