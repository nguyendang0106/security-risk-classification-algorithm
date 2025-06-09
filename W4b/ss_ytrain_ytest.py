import numpy as np

y_train = np.load("W4b/y_train.npy")
y_test = np.load("W4b/y_test.npy")

print(f" y_train shape: {y_train.shape}")
print(f" y_test shape: {y_test.shape}")

# Kiểm tra xem các nhãn có bị rò rỉ không
print(f" Nhãn trong train có trùng hoàn toàn test không? {set(y_test).issubset(set(y_train))}")

#  y_train shape: (2396,)
#  y_test shape: (394,)
#  Nhãn trong train có trùng hoàn toàn test không? True