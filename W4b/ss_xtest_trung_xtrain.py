import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

X_train = load_npz("W4b/X_train.npz")
X_test = load_npz("W4b/X_test.npz")

# Chuyển X_train, X_test sang dạng set để kiểm tra trùng lặp
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Kiểm tra số lượng mẫu test xuất hiện trong train
duplicate_count = sum(np.any(np.all(X_test_dense == x, axis=1)) for x in X_train_dense)

print(f" Số lượng mẫu test xuất hiện trong train: {duplicate_count}/{X_test.shape[0]}")


#  Số lượng mẫu test xuất hiện trong train: 12/394

# Nhận xét:

# Mặc dù chỉ có 12 mẫu bị trùng, nhưng vẫn có một mức độ rò rỉ dữ liệu.
# Điều này có thể dẫn đến mô hình "học vẹt", đặc biệt khi số mẫu huấn luyện không quá lớn.
# Giải pháp: Xóa 12 mẫu trùng hoặc thay đổi cách chia dữ liệu.