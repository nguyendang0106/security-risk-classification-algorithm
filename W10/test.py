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
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

# Giả sử bạn có module util.common hoặc dùng ConfusionMatrixDisplay
try:
    import W9.util.common as util
    PLOT_UTIL_AVAILABLE = True
except ImportError:
    print("Cảnh báo: Không tìm thấy module util.common. Sẽ sử dụng ConfusionMatrixDisplay của sklearn.")
    from sklearn.metrics import ConfusionMatrixDisplay
    PLOT_UTIL_AVAILABLE = False

# --- Định nghĩa các đường dẫn và hằng số ---
MODEL_DIR = pathlib.Path("W9/models1")
# TEST_DATA_PATH = pathlib.Path("W9/data2/clean/all_data.parquet") # CSE-CIC-IDS-2018
TEST_DATA_PATH = pathlib.Path("W9/data2/clean/all_data.parquet") # Đảm bảo đây là đường dẫn đúng tới CSE-CIC-IDS-2018
OUTPUT_DIR = pathlib.Path("W9/test_results_2018") # Thư mục lưu kết quả kiểm tra cho dataset 2018
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Các ngưỡng từ file train.py
# THRESHOLD_B = 0.7580776764761945 
# THRESHOLD_B = -0.0002196942507948895
THRESHOLD_B = -0.0004064190600459828
# THRESHOLD_M = 0.98  
THRESHOLD_M = 0.50            
# THRESHOLD_U = 0.0040588613744241275
# THRESHOLD_U = 0.004530129828299084
THRESHOLD_U = 0.0006590265510403005

# Các lớp dự đoán cuối cùng mà pipeline được huấn luyện để nhận diện + Unknown
FINAL_CLASSES = ['Benign', '(D)DOS', 'Botnet', 'Brute Force', 'Port Scan', 'Web Attack', 'Unknown']

# --- Tải dữ liệu kiểm tra (CSE-CIC-IDS-2018) ---
print(f"Đang tải dữ liệu kiểm tra từ: {TEST_DATA_PATH}")
try:
    test_df_full = pd.read_parquet(TEST_DATA_PATH) # Đổi tên biến
    print(f"Tải thành công {len(test_df_full)} mẫu.")

    # Lấy một phần dữ liệu để test
    N_SAMPLES_TO_TEST = 1000000 # Chọn số lượng mẫu bạn muốn thử
    test_df = test_df_full.head(N_SAMPLES_TO_TEST).copy() # Sử dụng .head() và .copy()
    print(f"*** Đang chạy thử nghiệm với {N_SAMPLES_TO_TEST} mẫu đầu tiên. ***")
    # !!! THÊM BƯỚC KIỂM TRA VÀ LOẠI BỎ CỘT KHÔNG CẦN THIẾT !!!
    columns_to_drop = []
    if 'Timestamp' in test_df.columns:
        columns_to_drop.append('Timestamp')
    # <<< THÊM DÒNG NÀY >>>
    if 'Destination Port' in test_df.columns:
         columns_to_drop.append('Destination Port')
    # <<< KẾT THÚC THÊM DÒNG >>>

    if columns_to_drop:
        test_df = test_df.drop(columns=columns_to_drop)
        print(f" - Đã loại bỏ các cột: {columns_to_drop}.")
    else:
        print(" - Không tìm thấy cột 'Timestamp' hoặc 'Destination Port' để loại bỏ.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu kiểm tra tại {TEST_DATA_PATH}")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file Parquet hoặc xử lý cột: {e}")
    exit()

# --- Tách features (X) và nhãn (y) ---
LABEL_COLUMN = 'Label' # Tên cột nhãn trong CSE-CIC-IDS-2018
if LABEL_COLUMN not in test_df.columns:
    print(f"Lỗi: Không tìm thấy cột nhãn '{LABEL_COLUMN}' trong file dữ liệu.")
    exit()

y_test_raw = test_df[LABEL_COLUMN]
X_test = test_df.drop(columns=[LABEL_COLUMN])
# !!! IN RA SỐ LƯỢNG FEATURES SAU KHI ĐÃ DROP TẤT CẢ CÁC CỘT KHÔNG MONG MUỐN !!!
print(f"Số lượng features sau khi loại bỏ Label và các cột khác: {X_test.shape[1]}")
print("Các nhãn gốc trong dữ liệu kiểm tra:")
print(y_test_raw.value_counts())

# --- Ánh xạ nhãn gốc (CSE-CIC-IDS-2018) sang các danh mục đã huấn luyện ---
print("\nÁnh xạ nhãn gốc sang danh mục huấn luyện...")

label_mapping = {
    # Benign
    'Benign': 'Benign',
    # (D)DOS
    'DDoS attacks-LOIC-HTTP': '(D)DOS',
    'DDOS attack-HOIC': '(D)DOS',
    'DoS attacks-Hulk': '(D)DOS',
    'DoS attacks-GoldenEye': '(D)DOS',
    'DoS attacks-Slowloris': '(D)DOS',
    'DDOS attack-LOIC-UDP': '(D)DOS',
    'DoS attacks-SlowHTTPTest': '(D)DOS',
    # Botnet
    'Bot': 'Botnet',
    # Brute Force
    'SSH-Bruteforce': 'Brute Force',
    'FTP-BruteForce': 'Brute Force',
    'Brute Force -Web': 'Brute Force', # Tấn công brute force vào web
    # Web Attack
    'Brute Force -XSS': 'Web Attack', # Bản chất là XSS, một loại tấn công web
    'SQL Injection': 'Web Attack',
    # Unknown (Các loại không có trong tập huấn luyện RF stage 2)
    'Infilteration': 'Unknown'
}

# Áp dụng ánh xạ, những nhãn không có trong mapping sẽ được giữ nguyên (cần kiểm tra sau)
y_test_mapped = y_test_raw.map(label_mapping).fillna(y_test_raw) # Giữ lại nhãn gốc nếu không map được

# Kiểm tra xem còn nhãn nào chưa được map vào FINAL_CLASSES không
unmapped_labels = set(y_test_mapped.unique()) - set(FINAL_CLASSES)
if unmapped_labels:
    print(f"Cảnh báo: Các nhãn sau không được ánh xạ vào FINAL_CLASSES: {unmapped_labels}")
    # Quyết định xử lý: có thể map chúng vào 'Unknown' hoặc báo lỗi tùy yêu cầu
    print("Mặc định sẽ coi các nhãn này là 'Unknown' để đánh giá.")
    y_test_final_comparison = y_test_mapped.apply(lambda x: x if x in FINAL_CLASSES else 'Unknown')
else:
    y_test_final_comparison = y_test_mapped

print("\nNhãn kiểm tra sau khi ánh xạ:")
print(y_test_final_comparison.value_counts())


# --- Tải Scalers và Models ---
print("\nĐang tải các Scaler và Model đã huấn luyện...")
try:
    with open(MODEL_DIR / "stage1_ocsvm_scaler.p", "rb") as f:
        scaler_stage1 = pickle.load(f)
        print(" - Tải thành công stage1_ocsvm_scaler.p")
    with open(MODEL_DIR / "stage2_rf_scaler.p", "rb") as f:
        scaler_stage2 = pickle.load(f)
        print(" - Tải thành công stage2_rf_scaler.p")
    with open(MODEL_DIR / "stage1_ocsvm.p", "rb") as f:
        ocsvm_pipeline = pickle.load(f)
        print(" - Tải thành công stage1_ocsvm.p (Pipeline)")
    with open(MODEL_DIR / "stage2_rf.p", "rb") as f:
        rf_model_stage2 = pickle.load(f)
        print(" - Tải thành công stage2_rf.p (RF với feature bổ sung)")
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file model hoặc scaler cần thiết: {e}")
    exit()
except Exception as e:
    print(f"Lỗi khi tải model/scaler: {e}")
    exit()

# --- Chuẩn hóa dữ liệu kiểm tra ---
print("\nĐang chuẩn hóa dữ liệu kiểm tra...")
# Bây giờ X_test đã có đúng số lượng cột (67) như scaler mong đợi
try:
    X_test_s1 = scaler_stage1.transform(X_test)
    print(f" - Dữ liệu Stage 1 scaled shape: {X_test_s1.shape}")
    X_test_s2 = scaler_stage2.transform(X_test)
    print(f" - Dữ liệu Stage 2 scaled shape: {X_test_s2.shape}")
except ValueError as e:
     # Lỗi này không nên xảy ra nữa nếu số cột đã khớp
     print(f"Lỗi khi transform dữ liệu: {e}")
     n_features_expected_s1 = scaler_stage1.n_features_in_
     n_features_expected_s2 = scaler_stage2.n_features_in_
     print(f"Scaler Stage 1 mong đợi {n_features_expected_s1} features, nhận được {X_test.shape[1]}.")
     print(f"Scaler Stage 2 mong đợi {n_features_expected_s2} features, nhận được {X_test.shape[1]}.")
     exit()


# --- Dự đoán Stage 1 (OCSVM) ---
print("\nThực hiện dự đoán Stage 1 (OCSVM)...")
anomaly_scores_s1 = -ocsvm_pipeline.decision_function(X_test_s1)
print(f" - Tính toán xong {len(anomaly_scores_s1)} điểm bất thường Stage 1.")

# --- Chuẩn bị dữ liệu đầu vào cho Stage 2 ---
print("\nChuẩn bị dữ liệu đầu vào cho Stage 2...")
X_test_s2_with_score = np.column_stack((X_test_s2, anomaly_scores_s1))
print(f" - Dữ liệu đầu vào Stage 2 (with score) shape: {X_test_s2_with_score.shape}")

# --- Dự đoán Stage 2 (Random Forest) ---
print("\nThực hiện dự đoán Stage 2 (Random Forest)...")
proba_s2 = rf_model_stage2.predict_proba(X_test_s2_with_score)
rf_classes = rf_model_stage2.classes_
print(f" - Tính toán xong {proba_s2.shape[0]} xác suất cho {proba_s2.shape[1]} lớp Stage 2.")
print(f" - Các lớp của mô hình RF Stage 2: {rf_classes}") # Phải khớp với các lớp tấn công đã huấn luyện + Unknown

# --- Áp dụng logic Multi-Stage Pipeline ---
print("\nÁp dụng logic phân loại đa tầng...")
# 1. Phân loại ban đầu Benign/Potential_Fraud
y_pred_temp = np.where(anomaly_scores_s1 < THRESHOLD_B, "Benign", "Potential_Fraud").astype(object)
print(f" - Stage 1 -> Benign/Potential_Fraud counts: {pd.Series(y_pred_temp).value_counts().to_dict()}")
# 2. Phân loại chi tiết/Unknown cho Potential_Fraud
fraud_indices = np.where(y_pred_temp == "Potential_Fraud")[0]
if len(fraud_indices) > 0:
    proba_s2_fraud = proba_s2[fraud_indices]
    max_proba_s2 = np.max(proba_s2_fraud, axis=1)
    pred_class_indices_s2 = np.argmax(proba_s2_fraud, axis=1)
    # Sử dụng rf_classes đã lấy từ mô hình RF đã tải
    pred_s2_specific = np.where(max_proba_s2 > THRESHOLD_M, rf_classes[pred_class_indices_s2], "Unknown")
    y_pred_temp[fraud_indices] = pred_s2_specific
    print(f" - Stage 2 -> Phân loại chi tiết/Unknown counts cho Potential_Fraud: {pd.Series(pred_s2_specific).value_counts().to_dict()}")
else:
    print(" - Stage 2 -> Không có mẫu nào là Potential_Fraud.")
# 3. Phân loại lại Unknown -> Benign/Unknown
unknown_indices = np.where(y_pred_temp == "Unknown")[0]
if len(unknown_indices) > 0:
    scores_s1_unknown = anomaly_scores_s1[unknown_indices]
    pred_extension = np.where(scores_s1_unknown < THRESHOLD_U, "Benign", "Unknown")
    y_pred_temp[unknown_indices] = pred_extension
    print(f" - Extension Stage -> Phân loại lại Unknown thành Benign/Unknown counts: {pd.Series(pred_extension).value_counts().to_dict()}")
else:
     print(" - Extension Stage -> Không có mẫu nào là Unknown.")

# Dự đoán cuối cùng
y_pred_final = y_pred_temp
print("\nPhân loại cuối cùng:")
print(pd.Series(y_pred_final).value_counts())

# --- Đánh giá hiệu suất ---
print("\nĐánh giá hiệu suất trên dữ liệu kiểm tra (CSE-CIC-IDS-2018 đã ánh xạ nhãn)...")

# Đảm bảo sử dụng tập nhãn đã ánh xạ để so sánh
present_labels_true = y_test_final_comparison.unique()
present_labels_pred = pd.Series(y_pred_final).unique()
all_labels_for_report = sorted(list(set(FINAL_CLASSES) | set(present_labels_true) | set(present_labels_pred)))

accuracy = accuracy_score(y_test_final_comparison, y_pred_final)
balanced_acc = balanced_accuracy_score(y_test_final_comparison, y_pred_final)
# Sử dụng labels=all_labels_for_report để đảm bảo tính toán trên tất cả các lớp có thể có
f1_macro = f1_score(y_test_final_comparison, y_pred_final, average='macro', labels=all_labels_for_report, zero_division=0)
f1_weighted = f1_score(y_test_final_comparison, y_pred_final, average='weighted', labels=all_labels_for_report, zero_division=0)

print(f"\nMetrics Tổng thể:")
print(f"  Accuracy:          {accuracy:.4f}")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  F1 Score (Macro):  {f1_macro:.4f}")
print(f"  F1 Score (Weight): {f1_weighted:.4f}")

print("\nClassification Report:")
# Quan trọng: Sử dụng y_test_final_comparison (nhãn đã map) làm y_true
print(classification_report(y_test_final_comparison, y_pred_final, labels=all_labels_for_report, zero_division=0))

# --- Vẽ ma trận nhầm lẫn ---
print("\nĐang vẽ ma trận nhầm lẫn...")
fig, ax = plt.subplots(figsize=(10, 8))

# Quan trọng: Sử dụng y_test_final_comparison làm y_true
cm_labels = all_labels_for_report # Các nhãn để hiển thị trên trục

if PLOT_UTIL_AVAILABLE:
    try:
        util.plot_confusion_matrix(
            y_test_final_comparison,
            y_pred_final,
            values=cm_labels, # Hiển thị các lớp này
            labels=cm_labels, # Sắp xếp theo các lớp này
            title="Confusion Matrix - Test Data (CSE-CIC-IDS-2018 Mapped)",
            ax=ax
        )
    except Exception as e:
        print(f"Lỗi khi dùng util.plot_confusion_matrix: {e}. Sử dụng ConfusionMatrixDisplay.")
        cm = confusion_matrix(y_test_final_comparison, y_pred_final, labels=cm_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot(ax=ax, xticks_rotation='vertical')
        ax.set_title("Confusion Matrix - Test Data (CSE-CIC-IDS-2018 Mapped)")
else:
    cm = confusion_matrix(y_test_final_comparison, y_pred_final, labels=cm_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix - Test Data (CSE-CIC-IDS-2018 Mapped)")

plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix_test_2018_mapped.png"
plt.savefig(cm_path)
print(f"Đã lưu ma trận nhầm lẫn vào: {cm_path}")
plt.show()

# --- Lưu kết quả dự đoán (Tùy chọn) ---
print("\nLưu kết quả dự đoán (tùy chọn)...")
results_df = pd.DataFrame({
    'true_label_raw': y_test_raw,          # Nhãn gốc từ file test
    'true_label_mapped': y_test_final_comparison, # Nhãn đã ánh xạ để đánh giá
    'predicted_label': y_pred_final,       # Nhãn dự đoán cuối cùng
    'stage1_anomaly_score': anomaly_scores_s1
})
for i, class_name in enumerate(rf_classes):
     results_df[f'stage2_proba_{class_name}'] = proba_s2[:, i]

results_path = OUTPUT_DIR / "test_predictions_2018_mapped.csv"
results_df.to_csv(results_path, index=False)
print(f"Đã lưu kết quả dự đoán chi tiết vào: {results_path}")

print("\nHoàn tất quá trình kiểm tra pipeline trên CSE-CIC-IDS-2018.")