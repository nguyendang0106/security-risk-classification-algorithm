from scipy.sparse import load_npz

X_train = load_npz("W4b/X_train.npz")
X_test = load_npz("W4b/X_test.npz")

print(f" X_train shape: {X_train.shape}")
print(f" X_test shape: {X_test.shape}")

# Kiểm tra số lượng đặc trưng của TF-IDF giữa train & test
print(f" Số đặc trưng TF-IDF giống nhau? {X_train.shape[1] == X_test.shape[1]}")

#  X_train shape: (2396, 5010)
#  X_test shape: (394, 5010)
#  Số đặc trưng TF-IDF giống nhau? True