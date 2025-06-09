# Seed value
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



# Load the models
f = open("W11/modelsComb/stage1_ocsvm_scaler.p","rb")
stage1_scaler = pickle.load(f)
f.close()
f = open("W11/modelsComb/stage2_rf_scaler.p","rb")
stage2_scaler = pickle.load(f)
f.close()


f = open("W11/modelsComb/stage1_ocsvm_100k.p","rb")
stage1 = pickle.load(f)
f.close()
f = open("W11/modelsComb/stage2_rf.p","rb")
stage2 = pickle.load(f)
f.close()

# Load the data
test = pd.read_parquet("W9/data1/test2/combined_mapped_data/combined_mapped_sampled_data.parquet")
# test = pd.read_csv("data/test.csv")

label_mapping_2017 = {
    'Benign': 'Benign',  
    'DoS Hulk':'(D)DOS', 
    'PortScan':'Port Scan', 
    'DDoS':'(D)DOS', 
    'DoS slowloris':'(D)DOS', 
    'DoS Slowhttptest':'(D)DOS', 
    'DoS GoldenEye':'(D)DOS', 
    'SSH-Patator':'Brute Force', 
    'FTP-Patator':'Brute Force', 
    'Bot': 'Botnet', 
    'Web Attack \x96 Brute Force': 'Web Attack', 
    'Web Attack \x96 Sql Injection': 'Web Attack', 
    'Web Attack \x96 XSS': 'Web Attack',
    # 'Infiltration': 'Infiltration',
    # 'Heartbleed': 'Heartbleed',
    'Infiltration': 'Unknown',
    'Heartbleed': 'Unknown',
    'Web Attack ï¿½ Brute Force': 'Web Attack',
    'Web Attack ï¿½ Sql Injection': 'Web Attack',
    'Web Attack ï¿½ XSS': 'Web Attack',
}

y_original_labels = test["Label"].copy()
y = test["Label"].apply(lambda label: label_mapping_2017.get(label, 'Unknown'))
x = test.drop(columns=['Label', 'Timestamp', 'Group'])

# Ensure x is a NumPy array for scalers if it's a DataFrame
x_np = x.values if isinstance(x, pd.DataFrame) else x

# Scale data for Stage 1
x_scaled_for_stage1 = stage1_scaler.transform(x_np)

sample = np.array(x_np[0]).reshape(1, -1)
print("Sample shape:", sample.shape)
print(sample)
print(y.value_counts())

tau_b = -0.10866126632226556
tau_m = 0.60
tau_u = -0.00015517209205490046

def hids_predict(x, tau_b, tau_m, tau_u):
    proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
    pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
    proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
    pred_2 = np.where(
        np.max(proba_2, axis=1) > tau_m, 
        stage2.classes_[np.argmax(proba_2, axis=1)], 
        "Unknown")
    proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
    pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
    pred_1[pred_1 == "Attack"] = pred_2
    pred_1[pred_1 == "Unknown"] = pred_3
    return pred_1

x_s_stage1_sample = stage1_scaler.transform(sample)
proba_stage1_sample = -stage1.decision_function(x_s_stage1_sample)
pred_stage1_sample = np.where(proba_stage1_sample < tau_b, "Benign", "Attack").astype(object)
np.unique(pred_stage1_sample, return_counts=True)


x_s_stage2_features_sample = stage2_scaler.transform(sample)
# Add the stage 1 score as an extra feature
input_for_stage2_sample = np.column_stack((x_s_stage2_features_sample, proba_stage1_sample.reshape(-1, 1)))

proba_stage2_sample = stage2.predict_proba(input_for_stage2_sample)
pred_2_sample = np.where(
    np.max(proba_stage2_sample, axis=1) > tau_m,
    stage2.classes_[np.argmax(proba_stage2_sample, axis=1)],
    "Unknown")
np.unique(pred_2_sample, return_counts=True)


#  Stage 1: Binary Detection
proba_1 = -stage1.decision_function(x_scaled_for_stage1) # Corrected: use scaled data
 # invert sign to act as anomaly score 
pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
np.unique(pred_1, return_counts=True)



# Stage 2: Multi-class Detection
attack_mask_stage1 = (pred_1 == "Attack")
x_attack_original_np = x_np[attack_mask_stage1] # Get original features of samples predicted as "Attack"

pred_2 = np.array([], dtype=object) # Initialize pred_2

if x_attack_original_np.shape[0] > 0: # Proceed only if there are "Attack" samples
    # Scale these original features using stage2_scaler
    x_attack_scaled_for_stage2 = stage2_scaler.transform(x_attack_original_np)
    
    # Get their corresponding Stage 1 scores (already calculated as proba_1)
    proba_1_for_attack_samples = proba_1[attack_mask_stage1]
    
    # Combine scaled features with Stage 1 scores
    input_for_stage2 = np.column_stack((x_attack_scaled_for_stage2, proba_1_for_attack_samples.reshape(-1, 1)))
    
    # proba_2 = stage2.predict_proba(x[pred_1 == "Attack"]) # Original incorrect line
    proba_2_raw = stage2.predict_proba(input_for_stage2) # Corrected
    pred_2 = np.where(
        np.max(proba_2_raw, axis=1) > tau_m,
        stage2.classes_[np.argmax(proba_2_raw, axis=1)],
        "Unknown"
    )
np.unique(pred_2, return_counts=True)




# Extension Stage: Zero-Day Detection
temp_pred_after_stage2 = pred_1.copy()
if x_attack_original_np.shape[0] > 0:
     temp_pred_after_stage2[attack_mask_stage1] = pred_2

unknown_after_stage2_mask = (temp_pred_after_stage2 == "Unknown")
proba_3_input_scores = proba_1[unknown_after_stage2_mask]

# proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"] # Original line, indexing can be tricky
pred_3 = np.where(proba_3_input_scores < tau_u, "Benign", "Unknown")
np.unique(pred_3, return_counts=True)




# Combine stages
y_pred = pred_1.copy()
if x_attack_original_np.shape[0] > 0:
    y_pred[attack_mask_stage1] = pred_2 # Apply stage 2 results

if np.any(unknown_after_stage2_mask): # Check if there are any "Unknown" to process for stage 3
    y_pred[unknown_after_stage2_mask] = pred_3 # Apply stage 3 results
np.unique(y_pred, return_counts=True)

def plot_confusion_matrix(y_true, y_pred, figsize=(7,7), cmap="Blues", values=[-1, 1], labels=["Attack", "Benign"], title="", ax=None, metrics=False):
    # Ensure y_true and y_pred are series/arrays for confusion_matrix
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    
    # Determine all unique labels present in either true or predicted, sorted
    # This ensures the confusion matrix handles all categories correctly.
    all_unique_labels = sorted(list(set(y_true_series.unique()) | set(y_pred_series.unique())))
    
    # If 'values' and 'labels' are passed, use them. Otherwise, use all_unique_labels.
    # For this specific problem, the passed 'labels' (classes) should be used.
    # However, ensure 'values' matches 'labels'.
    
    cm = confusion_matrix(y_true_series, y_pred_series, labels=labels) # Use the passed labels for order
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.zeros_like(cm, dtype=float) # Initialize with zeros
    # Avoid division by zero if a true class has no samples (cm_sum row is 0)
    valid_rows = cm_sum[:, 0] > 0
    cm_perc[valid_rows] = cm[valid_rows] / cm_sum[valid_rows].astype(float)
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
    cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, cmap=cmap, annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
    if title != "":
        ax.set_title(title)

# Xác định các lớp mà mô hình stage2 có thể dự đoán
stage2_predictable_classes = sorted(list(stage2.classes_))

# Tạo danh sách các danh mục cuối cùng để đánh giá
# Bao gồm 'Benign', 'Unknown', và các lớp từ stage2,
# nhưng loại trừ 'Port Scan' nếu nó có trong stage2.classes_ và bạn không muốn nó
final_evaluation_categories = ['Benign']
for cls in stage2_predictable_classes:
    final_evaluation_categories.append(cls)
final_evaluation_categories.append('Unknown')
final_evaluation_categories = sorted(list(set(final_evaluation_categories))) # Đảm bảo duy nhất và sắp xếp

# In ra để kiểm tra
print("Các danh mục sẽ được sử dụng để đánh giá:", final_evaluation_categories)

# Sử dụng final_evaluation_categories cho cả plot và report
plot_confusion_matrix(y, y_pred, labels=final_evaluation_categories, title="Confusion Matrix for 2017 Data")
plt.show()

print(classification_report(y, y_pred, digits=4, labels=final_evaluation_categories, zero_division=0))