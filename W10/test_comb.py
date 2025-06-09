import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import W9.util.common as util

def load_models(model_dir="W10/models_combined"):
    """Load all components of the combined model"""
    # Load models
    with open(f"{model_dir}/stage1_ocsvm.p", "rb") as f:
        stage1 = pickle.load(f)
    with open(f"{model_dir}/stage2_rf.p", "rb") as f:
        stage2 = pickle.load(f)
    
    # Load scalers
    with open(f"{model_dir}/stage1_ocsvm_scaler.p", "rb") as f:
        scaler1 = pickle.load(f)
    with open(f"{model_dir}/stage2_rf_scaler.p", "rb") as f:
        scaler2 = pickle.load(f)
    
    # Load thresholds
    with open(f"{model_dir}/optimal_thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)
    
    # Load feature indices
    with open(f"{model_dir}/selected_feature_indices.pkl", "rb") as f:
        feature_indices = pickle.load(f)
    
    return {
        'stage1': stage1,
        'stage2': stage2,
        'scaler1': scaler1,
        'scaler2': scaler2,
        'thresholds': thresholds,
        'feature_indices': feature_indices
    }

def predict(data, models):
    """Make predictions using the combined model"""
    # Apply feature selection
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, models['feature_indices']]
    else:
        data = data[:, models['feature_indices']]
    
    # Scale the data for stage 1
    data_scaled = models['scaler1'].transform(data)
    
    # Stage 1: Anomaly detection
    scores = -models['stage1'].decision_function(data_scaled)
    y_pred = np.where(scores < models['thresholds']['threshold_b'], "Benign", "Fraud").astype(object)
    
    # Prepare inputs for stage 2
    data_with_scores = np.column_stack((models['scaler2'].transform(data), scores))
    
    # Get indices for samples predicted as fraud
    fraud_indices = np.where(y_pred == "Fraud")[0]
    
    # Stage 2: Classification (only for samples classified as fraud)
    if len(fraud_indices) > 0:
        # Get probabilities from RF
        y_proba = models['stage2'].predict_proba(data_with_scores[fraud_indices])
        
        # Apply threshold
        max_proba = np.max(y_proba, axis=1)
        class_indices = np.argmax(y_proba, axis=1)
        
        # Replace "Fraud" with specific attack types or "Unknown"
        known_attacks = np.where(
            max_proba > models['thresholds']['threshold_m'],
            models['stage2'].classes_[class_indices],
            'Unknown'
        )
        y_pred[fraud_indices] = known_attacks
    
    # Extension: Reconsider "Unknown" predictions
    unknown_indices = np.where(y_pred == "Unknown")[0]
    if len(unknown_indices) > 0:
        # If anomaly score is low enough, classify as "Benign"
        y_pred[unknown_indices] = np.where(
            scores[unknown_indices] < models['thresholds']['threshold_u'],
            "Benign",
            "Unknown"
        )
    
    return y_pred

# Example usage
if __name__ == "__main__":
    # Load models
    models = load_models()
    
    # Load test data (change path as needed)
    test_data_path = "W10/data2/test_selected_features_2018.parquet"  # or any other dataset
    test_data = pd.read_parquet(test_data_path)
    X_test = test_data.drop(columns=['Label'])
    y_test = test_data['Label']
    
    # Map labels to common format if needed
    # (use the same mapping as in train_combined.py)
    
    # Make predictions
    y_pred = predict(X_test, models)
    
    # Evaluate
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    print("Test set metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
    util.plot_confusion_matrix(
        y_test,
        y_pred,
        values=classes,
        labels=classes,
        title="Model Prediction Results"
    )
    plt.tight_layout()
    plt.show()