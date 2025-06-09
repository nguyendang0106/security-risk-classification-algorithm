import pickle
from sklearn.metrics import classification_report

def test_model(model_path, X_test, y_test):
    """ Đánh giá mô hình """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_test)
    print(f" Kết quả đánh giá {model_path}:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Đọc dữ liệu từ file
    with open("W4/next/processed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Kiểm tra mô hình không có SMOTE
    test_model("W4/next/model_no_smote.pkl", X_test, y_test)

    # Kiểm tra mô hình có SMOTE
    test_model("W4/next/model_with_smote.pkl", X_test, y_test)


#      Kết quả đánh giá W4/next/model_no_smote.pkl:
#               precision    recall  f1-score   support

#         HIGH       0.69      0.81      0.74       200
#          LOW       0.46      0.14      0.21        44
#       MEDIUM       0.64      0.62      0.63       150

#     accuracy                           0.66       394
#    macro avg       0.59      0.52      0.53       394
# weighted avg       0.64      0.66      0.64       394

#  Kết quả đánh giá W4/next/model_with_smote.pkl:
#               precision    recall  f1-score   support

#         HIGH       0.74      0.72      0.73       200
#          LOW       0.50      0.66      0.57        44
#       MEDIUM       0.68      0.63      0.66       150

#     accuracy                           0.68       394
#    macro avg       0.64      0.67      0.65       394
# weighted avg       0.69      0.68      0.68       394


# Mô hình không dùng SMOTE (model_no_smote.pkl)
# Độ chính xác tổng thể: 66%
# Nhóm HIGH có recall 81%, tức là mô hình nhận diện tốt các mẫu HIGH.
# Nhóm LOW có recall 14%, rất thấp → Mô hình gần như bỏ sót các mẫu LOW.
# Nhóm MEDIUM có recall 62%, không quá tốt.
#  Nhận xét:
# Mô hình bị mất cân bằng. Nó dự đoán tốt nhóm HIGH, nhưng rất kém với nhóm LOW. Điều này là do dữ liệu ban đầu có ít mẫu LOW hơn, nên mô hình học thiên lệch về HIGH.



#  Mô hình có dùng SMOTE (model_with_smote.pkl)
# Độ chính xác tổng thể: 68% (cao hơn một chút so với mô hình trước)
# Nhóm HIGH giảm recall từ 81% → 72%, tức là bị giảm khả năng nhận diện HIGH.
# Nhóm LOW cải thiện recall từ 14% → 66%!
# → Đây là thay đổi đáng kể nhất: mô hình có thể nhận diện nhóm LOW tốt hơn.
# Nhóm MEDIUM gần như không thay đổi nhiều.



# Nhận xét:
# SMOTE giúp cải thiện nhận diện nhóm ít dữ liệu (LOW) nhưng ảnh hưởng đến nhóm HIGH. Tuy nhiên, tổng thể mô hình trở nên cân bằng hơn, tránh tình trạng bỏ sót các lỗ hổng bảo mật có độ nghiêm trọng thấp.



# Kết luận
# Dùng SMOTE tốt hơn trong trường hợp này vì:
# Cải thiện khả năng nhận diện nhóm LOW (tăng recall từ 14% → 66%).
# Độ chính xác tổng thể tăng nhẹ (66% → 68%).
# Mô hình trở nên cân bằng hơn, không quá ưu tiên HIGH như trước.