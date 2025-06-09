import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
from tqdm import tqdm
import time
import gc

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure results directory exists
os.makedirs('W6/results_tree', exist_ok=True)
os.makedirs('W6/models_tree', exist_ok=True) # Optional: If saving models

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

# Function to plot confusion matrix (similar to previous scripts)
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Plot and save confusion matrix"""
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"{title} (Counts)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f"{title} (Normalized)")
    plt.colorbar(im2, ax=ax2)

    for ax in [ax1, ax2]:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    thresh1 = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh1 else "black")
            ax2.text(j, i, f"{cm_normalized[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to '{filename}'")

# Function to train and evaluate a model
def train_evaluate_model(model_attack, model_risk, model_name,
                         X_train, y_train_attack, y_train_risk,
                         X_test, y_test_attack, y_test_risk,
                         attack_mapping, risk_mapping):
    """Trains and evaluates separate models for attack and risk classification."""
    print(f"\n=== Training and Evaluating {model_name} ===")
    results = {}

    # --- Attack Classification ---
    print(f"\nTraining {model_name} for Attack Classification...")
    start_time = time.time()
    model_attack.fit(X_train, y_train_attack)
    train_time_attack = (time.time() - start_time) / 60
    print(f"Attack model training time: {train_time_attack:.2f} minutes")

    print(f"Evaluating {model_name} for Attack Classification...")
    attack_preds = model_attack.predict(X_test)
    attack_acc = accuracy_score(y_test_attack, attack_preds) * 100
    attack_f1 = f1_score(y_test_attack, attack_preds, average='weighted', zero_division=0)

    print(f"\n{model_name} - Attack Classification Accuracy: {attack_acc:.2f}%")
    print(f"{model_name} - Attack Classification F1 Score: {attack_f1:.4f}")
    print(f"\n{model_name} - Attack Classification Report:")
    attack_class_names = [attack_mapping.get(i, f"Class_{i}") for i in range(len(attack_mapping))]
    print(classification_report(y_test_attack, attack_preds, labels=np.arange(len(attack_mapping)), target_names=attack_class_names, zero_division=0))

    plot_confusion_matrix(
        y_test_attack, attack_preds, attack_class_names,
        f"Attack Classification ({model_name})",
        f'W6/results_tree/attack_{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    )
    results['attack_acc'] = attack_acc
    results['attack_f1'] = attack_f1
    results['attack_train_time'] = train_time_attack

    # --- Risk Classification ---
    print(f"\nTraining {model_name} for Risk Classification...")
    start_time = time.time()
    model_risk.fit(X_train, y_train_risk)
    train_time_risk = (time.time() - start_time) / 60
    print(f"Risk model training time: {train_time_risk:.2f} minutes")

    print(f"Evaluating {model_name} for Risk Classification...")
    risk_preds = model_risk.predict(X_test)
    risk_acc = accuracy_score(y_test_risk, risk_preds) * 100
    risk_f1 = f1_score(y_test_risk, risk_preds, average='weighted', zero_division=0)

    print(f"\n{model_name} - Risk Classification Accuracy: {risk_acc:.2f}%")
    print(f"{model_name} - Risk Classification F1 Score: {risk_f1:.4f}")
    print(f"\n{model_name} - Risk Classification Report:")
    risk_class_names = [risk_mapping.get(i, f"Risk_{i}") for i in range(len(risk_mapping))]
    print(classification_report(y_test_risk, risk_preds, labels=np.arange(len(risk_mapping)), target_names=risk_class_names, zero_division=0))

    plot_confusion_matrix(
        y_test_risk, risk_preds, risk_class_names,
        f"Risk Classification ({model_name})",
        f'W6/results_tree/risk_{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    )
    results['risk_acc'] = risk_acc
    results['risk_f1'] = risk_f1
    results['risk_train_time'] = train_time_risk

    print("-" * 50)
    return results

# Main execution block
def main():
    print("=== Network Intrusion Detection: Tree-Based Models (RF, XGB, LGBM) ===")
    overall_start_time = time.time()

    # --- Data Loading and Preprocessing ---
    print("\nLoading datasets...")
    try:
        train_df = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test_df = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Store original test set indices if needed, though not strictly necessary here
    # test_indices = test_df.index

    combined_data = pd.concat([train_df, test_df], ignore_index=True).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")

    combined_data['risk_level'] = combined_data['attack_cat'].map(RISK_MAP).fillna('UNKNOWN')

    le_attack = LabelEncoder()
    le_risk = LabelEncoder()
    combined_data['attack_cat_encoded'] = le_attack.fit_transform(combined_data['attack_cat'])
    combined_data['risk_level_encoded'] = le_risk.fit_transform(combined_data['risk_level'])
    attack_mapping = {i: cat for i, cat in enumerate(le_attack.classes_)}
    risk_mapping = {i: level for i, level in enumerate(le_risk.classes_)}

    print("\nEncoding categorical features...")
    categorical_features = ['proto', 'service', 'state']
    for feature in categorical_features:
        if feature in combined_data.columns:
            combined_data[feature] = LabelEncoder().fit_transform(combined_data[feature].astype(str))

    features_to_drop = ['attack_cat', 'risk_level', 'attack_cat_encoded', 'risk_level_encoded']
    data_x = combined_data.drop(columns=features_to_drop, errors='ignore')
    data_y_attack = combined_data['attack_cat_encoded']
    data_y_risk = combined_data['risk_level_encoded']

    non_numeric_cols = data_x.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"Warning: Non-numeric columns remaining: {list(non_numeric_cols)}. Dropping them.")
        data_x = data_x.drop(columns=non_numeric_cols)

    # --- Train/Test Split (Crucial: Split before scaling!) ---
    # We use the original split provided by the dataset files
    train_size = len(train_df)
    X_train_raw = data_x[:train_size]
    X_test_raw = data_x[train_size:]
    y_train_attack = data_y_attack[:train_size].values
    y_test_attack = data_y_attack[train_size:].values
    y_train_risk = data_y_risk[:train_size].values
    y_test_risk = data_y_risk[train_size:].values

    print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    # --- Scaling (Fit on Train, Transform Train & Test) ---
    print("Standardizing features (fitting on training data only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw) # Use transform, not fit_transform

    # Clean up memory
    del train_df, test_df, combined_data, data_x, X_train_raw, X_test_raw
    gc.collect()

    # --- Imbalance Handling Setup ---
    # For XGBoost: Calculate scale_pos_weight for binary or use sample weights for multiclass
    # Since it's multiclass, we'll rely on the model's objective or use sample weights if needed.
    # For now, let's use the built-in capabilities where possible.

    # --- Model Initialization ---
    n_jobs = -1 # Use all available CPU cores
    random_state = 42

    # 1. Random Forest
    # Use class_weight='balanced' to handle imbalance
    rf_attack = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs, class_weight='balanced', max_depth=20, min_samples_split=5)
    rf_risk = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs, class_weight='balanced', max_depth=20, min_samples_split=5)

    # 2. XGBoost
    # For multiclass, objective 'multi:softmax' or 'multi:softprob' is used.
    # Imbalance handling can be done via 'scale_pos_weight' (binary only) or sample weights.
    # Let's start without explicit sample weights first. Use 'gpu_hist' if GPU is available and configured.
    xgb_attack = xgb.XGBClassifier(objective='multi:softmax', num_class=len(attack_mapping),
                                   eval_metric='mlogloss', use_label_encoder=False,
                                   random_state=random_state, n_estimators=150, learning_rate=0.1,
                                   max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs)
    xgb_risk = xgb.XGBClassifier(objective='multi:softmax', num_class=len(risk_mapping),
                                 eval_metric='mlogloss', use_label_encoder=False,
                                 random_state=random_state, n_estimators=150, learning_rate=0.1,
                                 max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs)

    # 3. LightGBM
    # Use class_weight='balanced'
    lgbm_attack = lgb.LGBMClassifier(objective='multiclass', num_class=len(attack_mapping),
                                     metric='multi_logloss', class_weight='balanced',
                                     random_state=random_state, n_estimators=150, learning_rate=0.1,
                                     num_leaves=40, max_depth=10, n_jobs=n_jobs)
    lgbm_risk = lgb.LGBMClassifier(objective='multiclass', num_class=len(risk_mapping),
                                   metric='multi_logloss', class_weight='balanced',
                                   random_state=random_state, n_estimators=150, learning_rate=0.1,
                                   num_leaves=40, max_depth=10, n_jobs=n_jobs)

    # --- Train and Evaluate Models ---
    all_results = {}
    all_results["Random Forest"] = train_evaluate_model(rf_attack, rf_risk, "Random Forest", X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk, attack_mapping, risk_mapping)
    all_results["XGBoost"] = train_evaluate_model(xgb_attack, xgb_risk, "XGBoost", X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk, attack_mapping, risk_mapping)
    all_results["LightGBM"] = train_evaluate_model(lgbm_attack, lgbm_risk, "LightGBM", X_train, y_train_attack, y_train_risk, X_test, y_test_attack, y_test_risk, attack_mapping, risk_mapping)

    # --- Final Summary ---
    print("\n===== FINAL RESULTS SUMMARY (Tree Models) =====")
    summary_data = []
    for model_name, metrics in all_results.items():
        summary_data.append([
            model_name,
            f"{metrics['attack_acc']:.2f}%",
            f"{metrics['attack_f1']:.4f}",
            f"{metrics['risk_acc']:.2f}%",
            f"{metrics['risk_f1']:.4f}",
            f"{metrics['attack_train_time'] + metrics['risk_train_time']:.2f} min"
        ])

    print(tabulate(summary_data, headers=["Model", "Attack Acc", "Attack F1", "Risk Acc", "Risk F1", "Total Train Time"]))
    print("-" * 80)

    print(f"\nTotal script execution time: {(time.time() - overall_start_time)/60:.2f} minutes")
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()