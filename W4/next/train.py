import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, use_smote=False):
    """ Huấn luyện mô hình """
    if use_smote:
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(" Số lượng mẫu sau SMOTE:\n", pd.Series(y_train).value_counts())

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Đọc dữ liệu từ file
    with open("W4/next/processed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Huấn luyện mô hình không dùng SMOTE
    model_no_smote = train_model(X_train, y_train, use_smote=False)
    with open("model_no_smote.pkl", "wb") as f:
        pickle.dump(model_no_smote, f)

    # Huấn luyện mô hình có dùng SMOTE
    model_with_smote = train_model(X_train, y_train, use_smote=True)
    with open("model_with_smote.pkl", "wb") as f:
        pickle.dump(model_with_smote, f)

    print(" Mô hình đã được huấn luyện và lưu lại!")


#      Số lượng mẫu sau SMOTE:
#  HIGH      799
# MEDIUM    799
# LOW       799
# Name: count, dtype: int64
#  Mô hình đã được huấn luyện và lưu lại!
