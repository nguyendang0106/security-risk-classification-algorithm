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

# Load Test Data
test = pd.read_parquet("W10/data2/test_selected_features_2018.parquet")
print("Before mapping")
print(test['Label'].value_counts())

# Mapping
test.loc[:, 'Label'] = test['Label'].replace([
    "DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC", "DoS attacks-Hulk", 
    "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "DDOS attack-LOIC-UDP", 
    "DoS attacks-SlowHTTPTest"
], "(D)DOS")

test.loc[:, 'Label'] = test['Label'].replace([
    "SSH-Bruteforce", "FTP-BruteForce", "Brute Force -Web"
], "Brute Force")

test.loc[:, 'Label'] = test['Label'].replace(["Bot"], "Botnet")
test.loc[:, 'Label'] = test['Label'].replace(["Brute Force -XSS", "SQL Injection"], "Web Attack")
test.loc[:, 'Label'] = test['Label'].replace(["Infilteration"], "Unknown")

print("After mapping")
print(test['Label'].value_counts())

# X, y
y = test['Label']
x = test.drop(columns=['Label'])
print(y.value_counts())

# Load models
f = open("W10/models1/stage1_ocsvm.p","rb")
stage1 = pickle.load(f)
f.close()
f = open("W10/models1/stage1_ocsvm_scaler.p","rb")
stage1_scaler = pickle.load(f)
f = open("W10/models1/stage2_rf.p","rb")
stage2 = pickle.load(f)
f.close()
f = open("W9/models1/stage2_rf_scaler.p","rb")
stage2_scaler = pickle.load(f)
f.close()


