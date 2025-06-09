# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# Add the project root directory to sys.path to allow finding W9
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

import pandas as pd
import W11.util.common as util
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import pathlib
import pickle
import matplotlib.pyplot as plt

# Import from skopt for Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, loguniform


from W9.util.AUROCEarlyStoppingPruneCallback import AUROCEarlyStoppingPruneCallback

# --- Create output directory ---
# output_dir = pathlib.Path("W11/modelsComb")
# output_dir.mkdir(parents=True, exist_ok=True)
# --- End create output directory ---

train = {
    "ocsvm": {}, # 10k samples
    "ae": {}, # 100k samples
    "stage2": {}
}
val = {
    "ocsvm": {},
    "ae": {},
    "stage2": {}
}
test = {
    # "y"
    # "y_binary"
    # "y_unknown"
    # "x"
}

#  Load Data Stage 1
clean_dir = "W9/data2/test/"


train["ocsvm"]["x"], train["ocsvm"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=10000, val_size=129485, test_size=56468)

val["ocsvm"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ocsvm"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))


train["ae"]["x"], train["ae"]["y"], x_benign_val, y_benign_val, _, _, x_malicious_train, y_malicious_train, _, _, _, _, _ = util.load_data(clean_dir, sample_size=1948,train_size=100000, val_size=129485, test_size=56468)

val["ae"]["x"] = np.concatenate((x_benign_val, x_malicious_train))
val["ae"]["y"] = np.concatenate((y_benign_val, np.full(y_malicious_train.shape[0], -1)))

#  Load Data Stage 2
n_benign_val = 1500

x_benign_train, _, _, _, x_benign_test, y_benign_test, x_malicious_train, y_malicious_train, x_malicious_test, y_malicious_test, attack_type_train, _, _ = util.load_data(clean_dir, sample_size=1948, train_size=n_benign_val, val_size=6815, test_size=56468)
train["stage2"]["x"], x_val, train["stage2"]["y"], y_val = train_test_split(x_malicious_train, y_malicious_train, stratify=attack_type_train, test_size=0.2, random_state=42, shuffle=True)

test['x'] = np.concatenate((x_benign_test, x_malicious_test))
test["y_n"] = np.concatenate((y_benign_test, np.full(y_malicious_test.shape[0], -1)))

val["stage2"]["x"] = np.concatenate((x_val, x_benign_train))
val["stage2"]["y"] = np.concatenate((y_val, np.full(n_benign_val, "Unknown")))

train["stage2"]["y_n"] = pd.get_dummies(train["stage2"]["y"])
val["stage2"]["y_n"] = pd.get_dummies(val["stage2"]["y"])

test["y"] = np.concatenate((np.full(56468, "Benign"), y_malicious_test))
test["y_unknown"] = np.where(
    (test["y"] == "Heartbleed") 
    | (test["y"] == "Infiltration") 
    | (test["y"] == "FTP-BruteForce")
                              , "Unknown", test["y"])
test["y_unknown_all"] = np.where(test['y_unknown'] == 'Benign', "Unknown", test['y_unknown'])

#  Scale the data
scaler = QuantileTransformer(output_distribution='normal')
train['ocsvm']['x_s'] = scaler.fit_transform(train['ocsvm']['x'])
val['ocsvm']['x_s'] = scaler.transform(val['ocsvm']['x'])
test['ocsvm_s'] = scaler.transform(test['x'])

scaler_ae = QuantileTransformer(output_distribution='normal')
train['ae']['x_s'] = scaler_ae.fit_transform(train['ae']['x'])
val['ae']['x_s'] = scaler_ae.transform(val['ae']['x'])
test['ae_s'] = scaler_ae.transform(test['x'])
# with open(output_dir / "stage1_ocsvm_scaler.p", "wb") as f:
#     pickle.dump(scaler_ae, f)
# print("Saved stage1_ocsvm_scaler.p")
# --- End Save Stage 1 Scaler ---

scaler_stage2 = QuantileTransformer(output_distribution='normal')
train['stage2']['x_s'] = scaler_stage2.fit_transform(train['stage2']['x'])
val['stage2']['x_s'] = scaler_stage2.transform(val['stage2']['x'])
test['stage2_s'] = scaler_stage2.transform(test['x'])
# with open(output_dir / "stage2_rf_scaler.p", "wb") as f:
#     pickle.dump(scaler_stage2, f)
# print("Saved stage2_rf_scaler.p")
# --- End Save Stage 2 Scaler ---

scaler = QuantileTransformer(output_distribution='uniform')
train['stage2']['x_q'] = scaler.fit_transform(train['stage2']['x'])
val['stage2']['x_q'] = scaler.transform(val['stage2']['x'])
test['stage2_q'] = scaler.transform(test['x'])

#  TRAIN MODELS
# STAGE 1: One-Class SVM
def create_ocsvm(params):
    return Pipeline(
        [
            # Add an imputer step to handle potential NaNs before PCA
            ("imputer", SimpleImputer(strategy='mean')), 
            ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
            ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
        ]
    ).set_params(**params)

# # Define the base pipeline structure (same as before)
# ocsvm_pipeline_base = Pipeline(
#     [
#         ("imputer", SimpleImputer(strategy='mean')), 
#         ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
#         ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
#     ]
# )
# # --- Define the parameter distributions to sample from --- 
# ocsvm_param_dist = {
#     'pca__n_components': randint(12,13), 
#     'ocsvm__gamma':loguniform(3e-2, 9e-2),
#     'ocsvm__nu': loguniform(1e-4, 3e-4), 
#     'ocsvm__kernel': ['rbf']
# }
# # --- End Define Distributions ---

# print("\n--- Optimizing Stage 1 OCSVM using RandomizedSearchCV ---") 
# n_iterations = 15
# clf_ocsvm = RandomizedSearchCV(
#     estimator=ocsvm_pipeline_base,
#     param_distributions=ocsvm_param_dist,
#     n_iter=n_iterations,
#     scoring='roc_auc',
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=seed_value,
# )

# # 10k
# print("Starting RandomizedSearchCV fit on 10k samples...")
# print(f"Starting RandomizedSearchCV fit ({n_iterations} iterations) on {train['ocsvm']['x_s'].shape[0]} benign samples...")
# if np.isnan(train['ocsvm']['y']).sum() > 0:
#     print("ERROR: NaNs found in train['ocsvm']['y']! This should not happen.")
# clf_ocsvm.fit(train['ocsvm']['x_s'], train['ocsvm']['y']) 
# print("\nRandomizedSearchCV for OCSVM finished.") 
# print("Best ROC AUC score on Validation set:", clf_ocsvm.best_score_)
# print("Best performing hyperparameters for OCSVM pipeline:")
# print(clf_ocsvm.best_params_)
# print(clf_ocsvm.best_estimator_)
# best_ocsvm_pipeline = clf_ocsvm.best_estimator_
# print("\nBest OCSVM Pipeline found:")
# print(best_ocsvm_pipeline)
# # --- Replace the manually trained ocsvm_model_10k ---
# ocsvm_model_10k = best_ocsvm_pipeline
# # --- End Replace ---

#  Train
params_ocsvm_10k = {
    "pca__n_components": 20,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0.045271231242100245,
    "ocsvm__nu": 0.0002841881747862226
}
print("Training OCSVM model (10k samples)...")
ocsvm_model_10k = create_ocsvm(params_ocsvm_10k)
# ocsvm_model_10k.fit(train['ocsvm']['x_s'])
print("OCSVM model (10k samples) trained.")
# with open(output_dir / "stage1_ocsvm_10k.p", "wb") as f:
#     pickle.dump(ocsvm_model_10k, f)
# print("Saved stage1_ocsvm_10k.p (Pipeline)")
f = open("W11/modelsComb/stage1_ocsvm_10k.p", "rb")
ocsvm_model_10k = pickle.load(f)
f.close()


# 100k
# print("Starting RandomizedSearchCV fit on 100k samples...") 
# print(f"Starting RandomizedSearchCV fit ({n_iterations} iterations) on {train['ae']['x_s'].shape[0]} benign samples...")
# if np.isnan(train['ae']['y']).sum() > 0:
#     print("ERROR: NaNs found in train['ae']['y']! This should not happen.")
# clf_ocsvm.fit(train['ae']['x_s'], train['ae']['y']) 
# print("\nRandomizedSearchCV for OCSVM finished.") 
# print("Best ROC AUC score on Validation set:", clf_ocsvm.best_score_)
# print("Best performing hyperparameters for OCSVM pipeline:")
# print(clf_ocsvm.best_params_)
# print(clf_ocsvm.best_estimator_)
# best_ocsvm_pipeline = clf_ocsvm.best_estimator_
# print("\nBest OCSVM Pipeline found:")
# print(best_ocsvm_pipeline)
# # --- Replace the manually trained ocsvm_model_100k ---
# ocsvm_model_100k = best_ocsvm_pipeline

# Train with equal training size as AE (100k)
params_ocsvm_100k = {
    "pca__n_components": 20,
    "ocsvm__kernel": "rbf",
    "ocsvm__gamma": 0.045271231242100245,
    "ocsvm__nu": 0.0002841881747862226
}
print("\nTraining OCSVM model (100k samples)...")
ocsvm_model_100k = create_ocsvm(params_ocsvm_100k)
# ocsvm_model_100k.fit(train['ae']['x_s'])
print("OCSVM model (100k samples) trained.")

# # --- Save Stage 1 OCSVM Pipeline and Model ---
# with open(output_dir / "stage1_ocsvm_100k.p", "wb") as f:
#     pickle.dump(ocsvm_model_100k, f)
# print("Saved stage1_ocsvm_100k.p (Pipeline)")
# --- End Save Stage 1 OCSVM ---
f = open("W11/modelsComb/stage1_ocsvm_100k.p", "rb")
ocsvm_model_100k = pickle.load(f)
f.close()

# Validation
print("\nValidating OCSVM model (10k)...")
score_val_10k = -ocsvm_model_10k.decision_function(val['ocsvm']['x_s'])
curves_metrics_10k, summary_metrics_10k = util.evaluate_proba(val['ocsvm']['y'], score_val_10k)
print("OCSVM (10k) Validation Summary:")
print(summary_metrics_10k)

print("\nValidating OCSVM model (100k)...")
score_val_100k = -ocsvm_model_100k.decision_function(val['ae']['x_s'])
curves_metrics_100k, summary_metrics_100k = util.evaluate_proba(val['ae']['y'], score_val_100k)
print("OCSVM (100k) Validation Summary:")
print(summary_metrics_100k)

#  Define Thresholds
quantiles = [0.995, 0.99, 0.975, 0.95, 0.935, 0.925, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
print("Thresholds based on 10k model validation:")
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics_10k.metric, summary_metrics_10k.FPR, summary_metrics_10k.threshold)})
print({q: np.quantile(score_val_10k[val["ocsvm"]["y"] == 1], q) for q in quantiles})

quantiles = [0.995, 0.99, 0.975, 0.95, 0.935, 0.925, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
print("\nThresholds based on 100k model validation:")
print({(metric, fpr): t for metric, fpr, t in zip(summary_metrics_100k.metric, summary_metrics_100k.FPR, summary_metrics_100k.threshold)})
print({q: np.quantile(score_val_100k[val["ae"]["y"] == 1], q) for q in quantiles})

#  Test
print("\nTesting OCSVM model (10k)...")
score_test_10k = -ocsvm_model_10k.decision_function(test['ocsvm_s'])
curves_metrics_test_10k, summary_metrics_test_10k = util.evaluate_proba(test["y_n"], score_test_10k)
# Print the summary
print("OCSVM (10k) Test Summary:")
print(summary_metrics_test_10k)


print("\nTesting OCSVM model (100k)...")
score_test_100k = -ocsvm_model_100k.decision_function(test['ae_s'])
curves_metrics_test_100k, summary_metrics_test_100k = util.evaluate_proba(test["y_n"], score_test_100k)
# Print the summary
print("OCSVM (100k) Test Summary:")
print(summary_metrics_test_100k)



# STAGE 2: Random Forest
def create_rf(params):
    return RandomForestClassifier(random_state=42).set_params(**params)

# # Train
# params = {
#     "n_estimators": 97,
#     "max_samples": 0.9034128710297624,
#     "max_features": 0.1751204590963604,
#     "min_samples_leaf": 1
# }
# # sota
params = {
          "class_weight": {'(D)DOS': np.float64(1.0),
                'Botnet': np.float64(1.0),
                'Brute Force': np.float64(1.0),
                'Port Scan': np.float64(1.0),
                'Unknown': np.float64(1.0),
                'Web Attack': np.float64(1.0)},
    "n_estimators": 75
}
# # Calculating class weights for balanced class weighted classifier training
# unique_classes = np.unique(train["stage2"]["y"])
# class_weights_array = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=unique_classes,
#     y=train["stage2"]["y"]
# )

# print("Unique classes found:", unique_classes)
# print("Calculated weights array:", class_weights_array)
# class_weights_dict = dict(zip(unique_classes, class_weights_array))
# print("Class weights dictionary:", class_weights_dict)

# model = RandomForestClassifier(
#     n_estimators=70,
#     criterion='gini',
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     min_weight_fraction_leaf=0.0,
#     max_features='sqrt',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     bootstrap=True,
#     oob_score=False,
#     n_jobs=None,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
#     class_weight=class_weights_dict,
#     ccp_alpha=0.0,
#     max_samples=None
# )

# hyperparameters = {
#     'n_estimators': [50, 75, 100]
# }

# clf = GridSearchCV(
#     estimator=model,
#     param_grid=hyperparameters,
#     cv=3,
#     verbose=1,
#     n_jobs=-1  
# )

# clf.fit(train['stage2']['x_s'], train["stage2"]["y"])

# print("Accuracy score on Validation set: \n")
# print(clf.best_score_ )
# print("---------------")
# print("Best performing hyperparameters on Validation set: ")
# print(clf.best_params_)
# print("---------------")
# print(clf.best_estimator_)
# rf_model_baseline = clf.best_estimator_ 
# print("\nUsing best estimator from GridSearchCV for Stage 2 Random Forest (Baseline)...")

print("\nTraining Stage 2 Random Forest (Baseline - without extra feature) (with class_weight='balanced')...")
rf_model_baseline = create_rf(params)
# rf_model_baseline.fit(train['stage2']['x_s'], train["stage2"]["y"])
print("Stage 2 Random Forest (Baseline) trained.")

# # --- Save Baseline RF ---
# with open(output_dir / "baseline_rf.p", "wb") as f:
#     pickle.dump(rf_model_baseline, f)
# print("Saved baseline_rf.p")
# # --- End Save Baseline RF ---
f = open("W11/modelsComb/baseline_rf.p", "rb")
rf_model_baseline = pickle.load(f)
f.close()

# Validation (Using Baseline RF)
print("\nValidating Stage 2 Random Forest (Baseline)...")
y_proba_val_2 = rf_model_baseline.predict_proba(val['stage2']['x_s']) 

# Define Thresholds
fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2, rf_model_baseline.classes_)
print(f_best["f1_weighted_threshold"])
y_pred_val_2 = np.where(np.max(y_proba_val_2, axis=1) > f_best["f1_weighted_threshold"], rf_model_baseline.classes_[np.argmax(y_proba_val_2, axis=1)], 'Unknown')

# Test (Using Baseline RF)
print("\nTesting Stage 2 Random Forest (Baseline)...")
y_proba_test_2 = rf_model_baseline.predict_proba(test['stage2_s'])
y_pred_test_2 = np.where(np.max(y_proba_test_2, axis=1) > f_best["f1_weighted_threshold"], rf_model_baseline.classes_[np.argmax(y_proba_test_2, axis=1)], 'Unknown')
print("Stage 2 Test Metrics (with thresholding):")
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_2, average='macro'),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_2, average='weighted'),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_2),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_2)
})

# Test Multi-Stage Model
# First Stage
# y_proba_1 = predictions["stage1"][3] # Using saved results from initial experiment
y_proba_1 = score_test_100k # See training ocsvm above

# Second threshold application - overwrites previous y_pred
threshold_b =  -0.13366126632226556 # F3 100k
y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud").astype(object)
print(np.unique(y_pred, return_counts=True))

# Second Stage
# y_proba_2 = predictions['stage2'][9] # Using saved results from initial experiment
y_proba_2 = y_proba_test_2 # See training rf above

threshold_m = 0.52 # See table above
y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, rf_model_baseline.classes_[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
print(np.unique(y_pred_2, return_counts=True))

# Combine first and second stage
y_pred[y_pred == "Fraud"] = y_pred_2
print(np.unique(y_pred, return_counts=True))

classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)
plt.show()

# Extension stage
threshold_u = -0.045517209205490046# 0.995 + 0.1 100k
y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
print(np.unique(y_pred_3, return_counts=True))

# Combine predictions 3 stages
y_pred[y_pred == "Unknown"] = y_pred_3
print(np.unique(y_pred, return_counts=True))

# Final Confusion Matrix
classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)
plt.show()

# Train second stage with anomaly score stage 1 as extra feature
proba_train = -ocsvm_model_100k.decision_function(train['stage2']['x_s'])
proba_val = -ocsvm_model_100k.decision_function(val['stage2']['x_s'])
proba_test = -ocsvm_model_100k.decision_function(test['stage2_s'])

print("Shapes:", proba_val.shape, val['stage2']['x_s'].shape, test['stage2_s'].shape)
train_with_proba = np.column_stack((train['stage2']['x_s'], proba_train))
val_with_proba = np.column_stack((val['stage2']['x_s'], proba_val))
test_with_proba = np.column_stack((test['stage2_s'], proba_test))
print("With proba shapes:", train_with_proba.shape, val_with_proba.shape, test_with_proba.shape)

# params = {
#     "n_estimators": 97,
#     "max_samples": 0.9034128710297624,
#     "max_features": 0.1751204590963604,
#     "min_samples_leaf": 1
# }
# sota
params = {
          "class_weight": {'(D)DOS': np.float64(1.0),
                'Botnet': np.float64(1.0),
                'Brute Force': np.float64(1.0),
                'Port Scan': np.float64(1.0),
                'Unknown': np.float64(1.0),
                'Web Attack': np.float64(1.0)},
    "n_estimators": 75
}
# rf_model = create_rf({})
print("\nTraining Stage 2 Random Forest (with extra feature)...")
rf_model_extra_feature = create_rf(params)
# rf_model_extra_feature.fit(train_with_proba, train["stage2"]["y"])
print("Stage 2 Random Forest (with extra feature) trained.")


# rf_model_extra_feature = RandomForestClassifier(
#     n_estimators=clf.best_params_['n_estimators'], 
#     class_weight=class_weights_dict,              
#     random_state=seed_value,                      
#     max_features='sqrt'
# )
# rf_model_extra_feature.fit(train_with_proba, train["stage2"]["y"])

# # --- Save Stage 2 RF (with extra feature) ---
# with open(output_dir / "stage2_rf.p", "wb") as f:
#     pickle.dump(rf_model_extra_feature, f)
# print("Saved stage2_rf.p")
# # --- End Save Stage 2 RF ---
f = open("W11/modelsComb/stage2_rf.p", "rb")
rf_model_extra_feature = pickle.load(f)
f.close()


y_proba_val_2_extra_feature = rf_model_extra_feature.predict_proba(val_with_proba) 
fmacro, fweight, threshold_fscore, f_best = util.optimal_fscore_multi(val['stage2']['y'], y_proba_val_2_extra_feature, rf_model_extra_feature.classes_)
print(f_best["f1_weighted_threshold"])
y_pred_val_2_extra_feature = np.where(np.max(y_proba_val_2_extra_feature, axis=1) > f_best["f1_weighted_threshold"], rf_model_extra_feature.classes_[np.argmax(y_proba_val_2_extra_feature, axis=1)], 'Unknown')

y_proba_test_2_extra_feature = rf_model_extra_feature.predict_proba(test_with_proba)
y_pred_test_2_extra_feature = np.where(np.max(y_proba_test_2_extra_feature, axis=1) > f_best["f1_weighted_threshold"], rf_model_extra_feature.classes_[np.argmax(y_proba_test_2_extra_feature, axis=1)], 'Unknown')
print({
    "f1_macro": f1_score(test["y_unknown_all"], y_pred_test_2_extra_feature, average='macro'),
    "f1_weighted": f1_score(test["y_unknown_all"], y_pred_test_2_extra_feature, average='weighted'),
    'accuracy': accuracy_score(test["y_unknown_all"], y_pred_test_2_extra_feature),
    'balanced_accuracy': balanced_accuracy_score(test["y_unknown_all"], y_pred_test_2_extra_feature)
})

# Full model performance
# y_proba_2 = predictions['stage2'][9] # Using saved results from initial experiment
y_proba_2 = y_proba_test_2_extra_feature # See training rf above

# --- Re-initialize y_pred using the Stage 1 results ---
threshold_b = 0.06309637046316713 # F1 100k
y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud").astype(object)
print(np.unique(y_pred, return_counts=True))

# threshold_m = 0.94 # See table above
threshold_m = 0.52 # See table above
y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, rf_model_extra_feature.classes_[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
print(np.unique(y_pred_2, return_counts=True))

y_pred[y_pred == "Fraud"] = y_pred_2
print(np.unique(y_pred, return_counts=True))

classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)
plt.show()

threshold_u = 0.15284100948918618 # 0.99 + 0.1 100k
y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
print(np.unique(y_pred_3, return_counts=True))

y_pred[y_pred == "Unknown"] = y_pred_3
print(np.unique(y_pred, return_counts=True))

# With default params
classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes)
plt.show()

def generateConfusionGraphs(y_proba_2_new, threshold_m, model_classes, include_metrics=False):
    fig, axs = plt.subplots(2,3, figsize=(18,12))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    y_proba_1 = score_test_100k
    metrics = []
    classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Web Attack', 'Unknown']
    
    y_pred_1_n = np.where(y_proba_1 < threshold_b, 1, -1)
    confusion_1_binary = util.plot_confusion_matrix(test['y_n'], y_pred_1_n, values=[1, -1], labels=["Benign", "Fraud"], title="Stage 1", ax=axs[0, 0])
    y_pred = np.where(y_proba_1 < threshold_b, "Benign", "Fraud")
    
    y_proba_2 = y_proba_2_new
    y_pred_2 = np.where(np.max(y_proba_2[y_pred == "Fraud"], axis=1) > threshold_m, model_classes[np.argmax(y_proba_2[y_pred == "Fraud"], axis=1)], 'Unknown')
    confusion_2_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred == "Fraud"], y_pred_2, values=classes, labels=classes, title="Stage 2", ax=axs[0, 1])

    y_pred = y_pred.astype(object)
    y_pred[y_pred == "Fraud"] = y_pred_2
    if include_metrics:
        result_12 = {
            "threshold_b": threshold_b,
            "threshold_m": threshold_m,
            "threshold_u": "-",
            "bACC": balanced_accuracy_score(test['y_unknown'], y_pred),
            "ACC": accuracy_score(test['y_unknown'], y_pred),
            "f1_micro": f1_score(test['y_unknown'], y_pred, average='micro'),
            "f1_macro": f1_score(test['y_unknown'], y_pred, average='macro'),
            "f1_weighted": f1_score(test['y_unknown'], y_pred, average='weighted'),
            "zero_day_recall_extension": "-",
            "zero_day_recall_total": "-"
        }
        metrics.append(result_12)
    confusion_12_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stage 1&2 Combined", ax=axs[0, 2])
    mask = ((y_pred == "Unknown") & (test['y_unknown_all'] == "Unknown"))
    
    y_pred_3 = np.where(y_proba_1[y_pred == "Unknown"] < threshold_u, "Benign", "Unknown")
    y_pred_3_n = np.where(y_proba_1[mask] < threshold_u, 1, -1)
    confusion_3_multi = util.plot_confusion_matrix(test['y_unknown'][y_pred == "Unknown"], y_pred_3, values=classes, labels=classes, title="Extension Multi-Class", ax=axs[1, 0])
    confusion_3_binary = util.plot_confusion_matrix(test['y_n'][mask], y_pred_3_n, values=[1, -1], labels=["Benign", "Zero-Day"], title="Extension Binary", ax=axs[1, 1])

    y_pred[y_pred == "Unknown"] = y_pred_3
    if include_metrics:
        result_123 = {
            "threshold_b": threshold_b,
            "threshold_m": threshold_m,
            "threshold_u": threshold_u,
            "bACC": balanced_accuracy_score(test['y_unknown'], y_pred),
            "ACC": accuracy_score(test['y_unknown'], y_pred),
            "f1_micro": f1_score(test['y_unknown'], y_pred, average='micro'),
            "f1_macro": f1_score(test['y_unknown'], y_pred, average='macro'),
            "f1_weighted": f1_score(test['y_unknown'], y_pred, average='weighted'),
            "zero_day_recall_extension": recall_score(test['y_n'][mask], y_pred_3_n, pos_label=-1),
            "zero_day_recall_total": (y_pred_3_n == -1).sum() / 47
        }
        metrics.append(result_123)
    confusion_123_multi = util.plot_confusion_matrix(test['y_unknown'], y_pred, values=classes, labels=classes, title="Stages 1,2 & Extension Combined", ax=axs[1, 2])
    return pd.DataFrame(metrics)

generateConfusionGraphs(y_proba_test_2_extra_feature, 0.94, rf_model_extra_feature.classes_, True) 
# plt.show()

# Default model params
generateConfusionGraphs(y_proba_test_2_extra_feature, 0.94, rf_model_extra_feature.classes_, True) 
# plt.show()

generateConfusionGraphs(y_proba_test_2_extra_feature, 0.94, rf_model_extra_feature.classes_, True) 
# plt.show()

generateConfusionGraphs(y_proba_test_2, 0.98, rf_model_baseline.classes_, True) 
# plt.show()

generateConfusionGraphs(y_proba_test_2, 0.52, rf_model_baseline.classes_, True) 
plt.show()  