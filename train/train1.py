# # Bạn có một DataFrame chứa thông tin khách hàng
# # Yêu cầu: Lọc các khách hàng có tuổi > 30 và đã mua hàng nhiều hơn 5 lần

# import pandas as pd

# df = pd.DataFrame({
#     'name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'age': [25, 35, 40, 22],
#     'purchases': [2, 7, 6, 1]
# })

# # Viết code để lọc ra các hàng thỏa điều kiện

# filter_df = df[(df['age'] > 30) & (df['purchases'] > 5)]
# print(filter_df)
# print( filter_df.shape[0] )  # In ra số lượng
# print()


# # Cho một mảng numpy gồm các số nguyên, viết hàm trả về số lượng phần tử là số chẵn

# import numpy as np
# arr = np.array([1, 4, 5, 8, 10, 13, 16])

# even_count = np.sum(arr % 2 == 0)
# print({even_count})

# # Viết hàm đếm tần suất xuất hiện của từng phần tử trong list

# from collections import Counter
# lst = ['a', 'b', 'a', 'c', 'b', 'a']
# cnt = Counter(lst)
# print(cnt)

# def is_palidrome(s):
#     return s == s[::-1]


# word = "radar"
# print(is_palidrome(word))

# # Cho list số nguyên, tìm phần tử lớn thứ 2 (distinct)

# lst = [3, 5, 1, 5, 2]


# def second_largest(lst):
#     unique_lst = list(set(lst))
#     unique_lst.sort()
#     return unique_lst[-2] if len(unique_lst) >= 2 else None


# print(second_largest(lst))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
