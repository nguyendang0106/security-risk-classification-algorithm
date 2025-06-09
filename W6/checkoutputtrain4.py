import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import gc

# ===================== Model Definition (Must match train4.py) =====================
class MultiTaskNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk):
        super(MultiTaskNeuralNet, self).__init__()
        self.shared_l1 = nn.Linear(input_size, hidden_size)
        self.shared_l2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.attack_out = nn.Linear(hidden_size_2, num_classes_attack)
        self.risk_out = nn.Linear(hidden_size_2, num_classes_risk)

    def forward(self, x):
        x = self.relu(self.shared_l1(x))
        x = self.relu(self.shared_l2(x))
        return self.attack_out(x), self.risk_out(x)

# ===================== Main Prediction Script =====================
def main():
    print("Loading datasets and preparing inputs...")
    data_dir = "W6/data"
    model_dir = "W6/models"

    try:
        # Ensure data directory exists
        if not os.path.exists(data_dir):
             print(f"Error: Data directory '{data_dir}' not found.")
             return
        train = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training-set.csv'))
        test = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_testing-set.csv'))
        train_rows = len(train)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- Replicate Preprocessing ---
    try:
        # Combine datasets
        combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)

        # Add severity_encoded column
        severity_map = {
            "Normal": 0, "Analysis": 2, "Backdoor": 3, "DoS": 3, "Exploits": 4,
            "Fuzzers": 1, "Generic": 2, "Reconnaissance": 1, "Shellcode": 4, "Worms": 4
        }
        combined_data['severity_encoded'] = combined_data['attack_cat'].map(severity_map)
        max_severity_value = combined_data['severity_encoded'].max() # Needed for model definition

        # Label encoding
        le_attack = LabelEncoder()
        combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat']) # Store encoded version
        num_attack_classes = len(le_attack.classes_) # Needed for model definition
        attack_class_names = list(le_attack.classes_) # Store names for mapping output

        combined_data['proto'] = LabelEncoder().fit_transform(combined_data['proto'])
        combined_data['service'] = LabelEncoder().fit_transform(combined_data['service'])
        combined_data['state'] = LabelEncoder().fit_transform(combined_data['state'])

        # Feature analysis for PCA
        lowSTD = list(combined_data.std(numeric_only=True).to_frame().nsmallest(7, columns=0).index)
        lowCORR = list(combined_data.corr(numeric_only=True).abs().sort_values('attack_cat_encoded')['attack_cat_encoded'].nsmallest(7).index)
        exclude = list(set(lowCORR + lowSTD))
        # Ensure target/original category columns are not in exclude list for PCA
        for col in ['attack_cat', 'attack_cat_encoded', 'severity_encoded']:
             if col in exclude:
                 exclude.remove(col)

        # Apply PCA
        pca = PCA(n_components=3)
        pca_input_data = combined_data[exclude].fillna(combined_data[exclude].mean())
        dim_reduct = pca.fit_transform(pca_input_data)
        combined_data.drop(exclude, axis=1, inplace=True)
        dim_reduction = pd.DataFrame(dim_reduct, columns=['PCA1', 'PCA2', 'PCA3'], index=combined_data.index)
        combined_data = combined_data.join(dim_reduction)

        # Scale 'dur'
        if 'dur' in combined_data.columns:
            combined_data['dur'] = 10000 * combined_data['dur']

        # Prepare data for modeling
        data_x = combined_data.drop(['attack_cat', 'attack_cat_encoded', 'severity_encoded'], axis=1)
        data_y_attack_encoded = combined_data['attack_cat_encoded'] # Need this for RFE split

        # Min-max scaling (Applied to the whole data_x in train4.py)
        scaler = MinMaxScaler()
        data_x_scaled = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns, index=data_x.index)

        # --- Get the first sample from the original test portion ---
        test_start_index = train_rows
        if test_start_index >= len(data_x_scaled):
            print(f"\nError: Cannot get test sample, index ({test_start_index}) is out of bounds.")
            return

        first_test_sample_all_features_series = data_x_scaled.iloc[test_start_index]
        input_data_model1 = first_test_sample_all_features_series.values # Numpy array (31 features)

        # --- Perform RFE to find selected features ---
        X_train_rfe_fit, _, y_train_attack_rfe_fit, _ = train_test_split(
            data_x_scaled, data_y_attack_encoded, test_size=0.20, random_state=42
        )
        rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10)
        rfe.fit(X_train_rfe_fit, y_train_attack_rfe_fit)
        selected_features = data_x_scaled.columns[rfe.support_]

        # --- Get input for Model 2 ---
        first_test_sample_rfe_features_series = first_test_sample_all_features_series[selected_features]
        input_data_model2 = first_test_sample_rfe_features_series.values # Numpy array (10 features)

        # Clean up intermediate dataframes
        del train, test, combined_data, data_x, data_x_scaled, X_train_rfe_fit
        gc.collect()

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Models and Predict ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- Model 1: All Features ---
    model1_path = os.path.join(model_dir, "multitask_nn_all_features.pth")
    print(f"\n--- Evaluating Model 1 (All Features) from {model1_path} ---")
    if not os.path.exists(model1_path):
        print("Model 1 file not found.")
    else:
        try:
            input_size_1 = input_data_model1.shape[0] # Should be 31
            model1 = MultiTaskNeuralNet(input_size_1, 128, 64, num_attack_classes, max_severity_value + 1)
            model1.load_state_dict(torch.load(model1_path, map_location=device))
            model1.to(device)
            model1.eval()

            # Prepare input tensor
            input_tensor1 = torch.tensor(input_data_model1, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out_attack1, out_risk1 = model1(input_tensor1)
                prob_attack1 = torch.softmax(out_attack1, dim=1)
                prob_risk1 = torch.softmax(out_risk1, dim=1)
                pred_idx_attack1 = torch.argmax(prob_attack1, dim=1).item()
                pred_idx_risk1 = torch.argmax(prob_risk1, dim=1).item()
                pred_name_attack1 = attack_class_names[pred_idx_attack1]

            print("Input shape:", input_tensor1.shape)
            print("Raw Attack Output (Logits):", out_attack1.cpu().numpy())
            print("Raw Risk Output (Logits):", out_risk1.cpu().numpy())
            print("\nPredicted Attack Index:", pred_idx_attack1)
            print("Predicted Attack Name:", pred_name_attack1)
            print("Predicted Risk Index:", pred_idx_risk1)
            # print("Attack Probabilities:", prob_attack1.cpu().numpy()) # Optional
            # print("Risk Probabilities:", prob_risk1.cpu().numpy()) # Optional

        except Exception as e:
            print(f"Error loading or predicting with Model 1: {e}")
            import traceback
            traceback.print_exc()

    # --- Model 2: RFE Features ---
    model2_path = os.path.join(model_dir, "multitask_nn_selected_features.pth")
    print(f"\n--- Evaluating Model 2 (RFE Features) from {model2_path} ---")
    if not os.path.exists(model2_path):
        print("Model 2 file not found.")
    else:
        try:
            input_size_2 = input_data_model2.shape[0] # Should be 10
            model2 = MultiTaskNeuralNet(input_size_2, 128, 64, num_attack_classes, max_severity_value + 1)
            model2.load_state_dict(torch.load(model2_path, map_location=device))
            model2.to(device)
            model2.eval()

            # Prepare input tensor
            input_tensor2 = torch.tensor(input_data_model2, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out_attack2, out_risk2 = model2(input_tensor2)
                prob_attack2 = torch.softmax(out_attack2, dim=1)
                prob_risk2 = torch.softmax(out_risk2, dim=1)
                pred_idx_attack2 = torch.argmax(prob_attack2, dim=1).item()
                pred_idx_risk2 = torch.argmax(prob_risk2, dim=1).item()
                pred_name_attack2 = attack_class_names[pred_idx_attack2]

            print("Input shape:", input_tensor2.shape)
            print("Raw Attack Output (Logits):", out_attack2.cpu().numpy())
            print("Raw Risk Output (Logits):", out_risk2.cpu().numpy())
            print("\nPredicted Attack Index:", pred_idx_attack2)
            print("Predicted Attack Name:", pred_name_attack2)
            print("Predicted Risk Index:", pred_idx_risk2)
            # print("Attack Probabilities:", prob_attack2.cpu().numpy()) # Optional
            # print("Risk Probabilities:", prob_risk2.cpu().numpy()) # Optional

        except Exception as e:
            print(f"Error loading or predicting with Model 2: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()