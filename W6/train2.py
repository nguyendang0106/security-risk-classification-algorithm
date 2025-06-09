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

# PyTorch imports
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore")

import faulthandler
faulthandler.enable()

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, label, output_dir):
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Accuracy of {label}: {acc:.7f}")
    joblib.dump(model, os.path.join(output_dir, f"{label}.pkl"))
    return acc

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
            # First round: train with all features
            print("\n--- Training XGB & LGBM with All Features ---")
            models = [
                (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 'XGB_attack_all', y_train_attack, y_test_attack),
                (LGBMClassifier(), 'LGBM_attack_all', y_train_attack, y_test_attack),
                (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 'XGB_severity_all', y_train_severity, y_test_severity),
                (LGBMClassifier(), 'LGBM_severity_all', y_train_severity, y_test_severity)
            ]

            for model, label, y_tr, y_te in models:
                train_and_evaluate_model(model, X_train, X_test, y_tr, y_te, label, "W6/models")
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

            # Second round: train with selected features
            print("\n--- Training XGB & LGBM with Selected Features ---")
            models_rfe = [
                (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 'XGB_attack_rfe', y_train_attack, y_test_attack),
                (LGBMClassifier(), 'LGBM_attack_rfe', y_train_attack, y_test_attack),
                (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 'XGB_severity_rfe', y_train_severity, y_test_severity),
                (LGBMClassifier(), 'LGBM_severity_rfe', y_train_severity, y_test_severity)
            ]

            for model, label, y_tr, y_te in models_rfe:
                train_and_evaluate_model(model, X_train_RFE, X_test_RFE, y_tr, y_te, label, "W6/models")


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