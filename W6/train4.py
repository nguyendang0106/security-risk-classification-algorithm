import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import os
import collections
from tabulate import tabulate

# Scikit-learn imports
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

# ===================== Dataset Class =====================
class UNSWDataset(Dataset):
    def __init__(self, x, y_attack, y_risk):
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y_attack = torch.tensor(y_attack.values, dtype=torch.long)
        self.y_risk = torch.tensor(y_risk.values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_attack[idx], self.y_risk[idx]

# ===================== Multi-Task Neural Network =====================
class MultiTaskNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk):
        super(MultiTaskNeuralNet, self).__init__()
        self.shared_l1 = nn.Linear(input_size, hidden_size)
        self.shared_l2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()

        # Output heads
        self.attack_out = nn.Linear(hidden_size_2, num_classes_attack)
        self.risk_out = nn.Linear(hidden_size_2, num_classes_risk)

    def forward(self, x):
        x = self.relu(self.shared_l1(x))
        x = self.relu(self.shared_l2(x))
        return self.attack_out(x), self.risk_out(x)

# ===================== Training Function =====================
def train_neural_network(model, train_loader, val_loader, device, epochs=40, lr=0.001):
    criterion_attack = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_attack = 0
        correct_risk = 0
        total = 0

        for x, y_attack, y_risk in train_loader:
            x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)
            optimizer.zero_grad()

            out_attack, out_risk = model(x)
            loss_attack = criterion_attack(out_attack, y_attack)
            loss_risk = criterion_risk(out_risk, y_risk)
            loss = loss_attack + loss_risk

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred_attack = torch.max(out_attack, 1)
            _, pred_risk = torch.max(out_risk, 1)
            correct_attack += (pred_attack == y_attack).sum().item()
            correct_risk += (pred_risk == y_risk).sum().item()
            total += y_attack.size(0)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.7f}, Attack Acc: {correct_attack/total:.7f}, Risk Acc: {correct_risk/total:.7f}")

    # Validation
    model.eval()
    val_correct_attack = 0
    val_correct_risk = 0
    val_total = 0
    with torch.no_grad():
        for x, y_attack, y_risk in val_loader:
            x, y_attack, y_risk = x.to(device), y_attack.to(device), y_risk.to(device)
            out_attack, out_risk = model(x)
            _, pred_attack = torch.max(out_attack, 1)
            _, pred_risk = torch.max(out_risk, 1)
            val_correct_attack += (pred_attack == y_attack).sum().item()
            val_correct_risk += (pred_risk == y_risk).sum().item()
            val_total += y_attack.size(0)

    print(f"Validation Attack Accuracy: {val_correct_attack/val_total:.7f}, Risk Accuracy: {val_correct_risk/val_total:.7f}")

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

    try:
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
        print("CORRECT Attack LabelEncoder Order:", list(le.classes_))


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

        # Store accuracies
        accuracies = {}

        try:
            # ===================== LẦN 1: TRAIN VỚI TOÀN BỘ ĐẶC TRƯNG =====================
            print("\n================== Train with ALL features ==================")
            scaler = MinMaxScaler()
            data_x_scaled = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)

            x_train, x_val, y_train_attack, y_val_attack, y_train_risk, y_val_risk = train_test_split(
                data_x_scaled, data_y_attack, data_y_severity, test_size=0.2, random_state=42)

            train_dataset = UNSWDataset(x_train, y_train_attack, y_train_risk)
            val_dataset = UNSWDataset(x_val, y_val_attack, y_val_risk)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

            input_size = x_train.shape[1]
            model = MultiTaskNeuralNet(input_size, 128, 64, len(le.classes_), max(data_y_severity) + 1)
            train_neural_network(model, train_loader, val_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), epochs=40)
            torch.save(model.state_dict(), "W6/models/multitask_nn_all_features.pth")
            print("Saved model with all features to: W6/models/multitask_nn_all_features.pth")
        except Exception as e:
            print(f"Error during model training: {e}")
# -----------------------
        # Feature selection with RFE
        print("\n--- Feature Selection with RFE ---")
        try:
            rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10).fit(X_train, y_train_attack)
            desiredIndices = np.where(rfe.support_==True)[0]
            selected_features = list(X_train.columns[desiredIndices])
            print("Selected features:", selected_features)

            # Convert to DataFrame if not already
            X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
            X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

            whitelist = X_train_df.columns.values[desiredIndices]
            X_train_RFE, X_test_RFE = X_train_df[whitelist], X_test_df[whitelist]
            
            print(f"RFE selected {len(whitelist)} features")
            print(f"X_train_RFE shape: {X_train_RFE.shape}, X_test_RFE shape: {X_test_RFE.shape}")

            # ===================== LẦN 2: TRAIN VỚI FEATURE ĐÃ CHỌN =====================
            print("\n================== Train with SELECTED features ==================")

            data_x_selected = data_x[selected_features]
            data_x_selected = pd.DataFrame(scaler.fit_transform(data_x_selected), columns=data_x_selected.columns)

            x_train, x_val, y_train_attack, y_val_attack, y_train_risk, y_val_risk = train_test_split(
                data_x_selected, data_y_attack, data_y_severity, test_size=0.2, random_state=42)

            train_dataset = UNSWDataset(x_train, y_train_attack, y_train_risk)
            val_dataset = UNSWDataset(x_val, y_val_attack, y_val_risk)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

            input_size = x_train.shape[1]
            model = MultiTaskNeuralNet(input_size, 128, 64, len(le.classes_), max(data_y_severity) + 1)
            train_neural_network(model, train_loader, val_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), epochs=40)
            torch.save(model.state_dict(), "W6/models/multitask_nn_selected_features.pth")
            print("Saved model with selected features to: W6/models/multitask_nn_selected_features.pth")


            print("Sample y_train_attack:", y_train_attack[:10].values)
            print("Sample y_train_severity:", y_train_severity[:10].values)
            print("Classes attack:", set(y_train_attack))
            print("Classes severity:", set(y_train_severity))
            
            # Print summary of all models
            print("\n--- Model Accuracy Summary ---")
            for model_name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
                print(f"{model_name:25s}: {accuracy:.7f}")
            
        except Exception as e:
            print(f"Error during model training: {e}")
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()