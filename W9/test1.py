# # Seed value
# seed_value= 42

# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)

# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

# # 3. Set the `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)

# import pandas as pd
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.metrics import auc, roc_curve, accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.svm import OneClassSVM
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# #  Load Test Data
# test = pd.read_parquet("W10/data2/test.parquet")
# # test = pd.read_csv("data/test.csv")
# print(test["Y"].value_counts())

# y = test["Y"].replace(["Heartbleed", "Infiltration"], "Unknown")
# x = test.drop(columns=['Y'])

# print(y.value_counts())

# #  Load additional infiltration samples from 2018
# infiltration_2018 = pd.read_parquet("W10/data2/infiltration_2018.parquet")

# y_18 = infiltration_2018['y']
# x_18 = infiltration_2018.drop(columns=['y'])

# print(y_18.value_counts())

# # Load Models
# # the pipelines with feature scaler and optimized model combined for binary detection and multi-class classification
# # the individual feature scalers and optimized models
# # Random Forest (RF) optimized baseline model and feature scaler
# # Optimized models following Bovenzi et al. for comparitative analysis

# # Optimized pipelines
# f = open("W9/models1/stage1_ocsvm.p","rb")
# stage1 = pickle.load(f)
# f.close()
# f = open("W9/models1/stage2_rf.p","rb")
# stage2 = pickle.load(f)
# f.close()

# # Individual feature scalers and classification models
# f = open("W9/models1/stage1_ocsvm_model.p","rb")
# stage1_model = pickle.load(f)
# f.close()
# f = open("W9/models1/stage1_ocsvm_scaler.p","rb")
# stage1_scaler = pickle.load(f)
# f.close()
# f = open("W9/models1/stage2_rf_model.p","rb")
# stage2_model = pickle.load(f)
# f.close()
# f = open("W9/models1/stage2_rf_scaler.p","rb")
# stage2_scaler = pickle.load(f)
# f.close()

# # RF baseline model and feature scaler
# f = open("W9/models1/baseline_rf.p","rb")
# baseline_rf = pickle.load(f)
# f.close()
# f = open("W9/models1/baseline_rf_scaler.p","rb")
# baseline_rf_scaler = pickle.load(f)
# f.close()

# # Optimized models for Bovenzi et al.
# from tensorflow import keras
# sota_stage1 = keras.models.load_model("W9/models1/sota_stage1.h5")
# f = open("W9/models1/sota_stage2.p","rb")
# sota_stage2 = pickle.load(f)
# f.close()

# #  Thresholds
# tau_b = -0.0002196942507948895
# tau_m = 0.98
# tau_u = 0.0040588613744241275

# #  Evaluation of Time Complexity
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning) 

# # hids_predict: Function to preform classification of all stages combined of the novel hierarchical 
# # multi-stage intrusion detection approach by Verkerken et al.

# # hids_sota_predict: Function to evaluate former SotA approach existing of two stages by Bovenzi et al.


# def hids_predict(x, tau_b, tau_m, tau_u):
#     proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
#     pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
#     proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
#     pred_2 = np.where(
#         np.max(proba_2, axis=1) > tau_m, 
#         stage2.classes_[np.argmax(proba_2, axis=1)], 
#         "Unknown")
#     proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
#     pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
#     pred_1[pred_1 == "Attack"] = pred_2
#     pred_1[pred_1 == "Unknown"] = pred_3
#     return pred_1

# def hids_sota_predict(x, tau_b, tau_m):
#     x_s = stage1_scaler.transform(x)
#     x_pred = sota_stage1.predict(x_s)
#     proba_1 = np.sum((x_s - x_pred)**2, axis=1)
#     pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
#     x_s = stage2_scaler.transform(x)
#     proba_2 = sota_stage2.predict_proba(x_s[pred_1 == "Attack"])
#     pred_1[pred_1 == "Attack"] = np.where(
#         np.max(proba_2, axis=1) > tau_m, 
#         stage2.classes_[np.argmax(proba_2, axis=1)], 
#         "Unknown")
#     return pred_1

# # Max F-score thesholds
# %%timeit -r3 -n3 -p6
# tau_b = -0.0002196942507948895
# tau_m = 0.98
# tau_u = 0.004530129828299084
# y = hids_predict(x, tau_b, tau_m, tau_u)

# # Max bACC thresholds
# %%timeit -r3 -n3 -p6
# tau_b = -0.0004064190600459828
# tau_m = 0.98
# tau_u = 0.0006590265510403005
# y = hids_predict(x, tau_b, tau_m, tau_u)

# # Best "balanced" thesholds
# %%timeit -r3 -n3 -p6
# tau_b = -0.0002196942507948895
# tau_m = 0.98
# tau_u = 0.0040588613744241275
# y = hids_predict(x, tau_b, tau_m, tau_u) 

# # Baseline RF
# threshold = 0.43 

# %%timeit -r3 -n3 -p6
# x_s = baseline_rf_scaler.transform(x)
# y_proba = baseline_rf.predict_proba(x_s)
# y_pred = np.where(np.max(y_proba, axis=1) > threshold, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')

# #  Bovenzi et al.
# # Thresholds experimentally optimized
# tau_b = 0.7580776764761945
# tau_m = 0.98

# %%timeit -r3 -n3 -p6
# y = hids_sota_predict(x, tau_b, tau_m)

# #  Single sample
# # Inference time for predicting a single flow
# sample = np.array(x.values[0]).reshape(1, -1)
# print(sample)

#  %%timeit -n 10 -r 10
# x_s = stage1_scaler.transform(sample)
# proba = -stage1_model.decision_function(x_s)
# pred = np.where(proba < tau_b, "Benign", "Attack").astype(object)

#  %%timeit -n 10 -r 10
# x_s = stage2_scaler.transform(sample)
# proba = stage2_model.predict_proba(x_s)
# pred_2 = np.where(
#     np.max(proba, axis=1) > tau_m, 
#     stage2_model.classes_[np.argmax(proba, axis=1)], 
#     "Unknown")

# %%timeit -n 10 -r 10
# proba_1 = -stage1.decision_function(sample) # invert sign to act as anomaly score 
# pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)

# %%timeit -n 10 -r 10
# proba_2 = stage2.predict_proba(sample)
# pred_2 = np.where(
#     np.max(proba_2, axis=1) > tau_m, 
#     stage2.classes_[np.argmax(proba_2, axis=1)], 
#     "Unknown")

# # Evaluate Multi-Stage Model

# #  Stage 1: Binary Detection
# proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
# pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
# print(np.unique(pred_1, return_counts=True))

# %%timeit -n3 -r3
# proba_1 = -stage1.decision_function(x) # invert sign to act as anomaly score 
# pred_1 = np.where(proba_1 < tau_b, "Benign", "Attack").astype(object)
# print(np.unique(pred_1, return_counts=True))

# # Stage 2: Multi-Class Classification
# proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
# pred_2 = np.where(
#     np.max(proba_2, axis=1) > tau_m, 
#     stage2.classes_[np.argmax(proba_2, axis=1)], 
#     "Unknown")
# print(np.unique(pred_2, return_counts=True))

# %%timeit -n3 -r3
# proba_2 = stage2.predict_proba(x[pred_1 == "Attack"])
# pred_2 = np.where(
#     np.max(proba_2, axis=1) > tau_m, 
#     stage2.classes_[np.argmax(proba_2, axis=1)], 
#     "Unknown")
# print(np.unique(pred_2, return_counts=True))

# # Extension Stage: Zero-Day Detection
# proba_3 = proba_1[pred_1 == "Attack"][pred_2 == "Unknown"]
# pred_3 = np.where(proba_3 < tau_u, "Benign", "Unknown")
# print(np.unique(pred_3, return_counts=True))

# # Combine stages
# y_pred = pred_1.copy()
# y_pred[y_pred == "Attack"] = pred_2
# y_pred[y_pred == "Unknown"] = pred_3
# print(np.unique(y_pred, return_counts=True))


# #  Statistics and Visualizations of the Results
# def plot_confusion_matrix(y_true, y_pred, figsize=(7,7), cmap="Blues", values=[-1, 1], labels=["Attack", "Benign"], title="", ax=None, metrics=False):
#     cm = confusion_matrix(y_true, y_pred, labels=values)
#     cm_sum = np.sum(cm, axis=1, keepdims=True)
#     cm_perc = cm / cm_sum.astype(float)
#     annot = np.empty_like(cm).astype(str)
#     nrows, ncols = cm.shape
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
#     cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
#     cm_perc.index.name = 'Actual'
#     cm_perc.columns.name = 'Predicted'
#     if ax == None:
#         fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(cm_perc, cmap=cmap, annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
#     if title != "":
#         ax.set_title(title)

# classes = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']
# plot_confusion_matrix(y, y_pred, values=classes, labels=classes, metrics=True)
# plt.show()

# print(classification_report(y, y_pred, digits=4)) 

# # Robustness - Preform classification on additional "infiltration" samples from cic-ids-2018
# tau_b = -0.0002196942507948895
# tau_m = 0.98
# tau_u = 0.0040588613744241275 # balanced threshold -> 29,02% recall on infiltration 2018
# # tau_u = 0.0006590265510403005 # bACC threshold -> 78,38% recall on infiltration 2018
# y = hids_predict(x_18, tau_b, tau_m, tau_u)
# print(np.unique(y, return_counts=True))

# tau_b = 0.7580776764761945
# tau_m = 0.98
# y = hids_sota_predict(x_18, tau_b, tau_m) # 86.99% recall on infiltration 2018
# print(np.unique(y, return_counts=True))


# x_s = baseline_rf_scaler.transform(x_18)
# y_proba = baseline_rf.predict_proba(x_s)
# y_pred = np.where(np.max(y_proba, axis=1) > 0.43, baseline_rf.classes_[np.argmax(y_proba, axis=1)], 'Unknown')
# # 0.06% recall on infiltration 2018
# print(np.unique(y_pred, return_counts=True))