import pandas as pd


#access files
file1 = pd.read_csv('W3/dataset2/products.csv')
file2 = pd.read_csv('W3/dataset2/vendors.csv')
file3 = pd.read_csv('W3/dataset2/cve.csv')
file4 = pd.read_csv('W3/dataset2/vendor_product.csv')

print('product: \n', file1.head(), len(file1))
print('\nvendors: \n', file2.head(), len(file2))
print('\ncve: \n',file3.head(), len(file3))
print('\nvendor_product: \n', file4.head(), len(file4))

# Nhìn vào giá trị null trong khung dữ liệu:
print('file1: \n', file1.info())  # một số NaN trong cột vulnerability_product -- một số cv_id không liên kết với bất kỳ sản phẩm nào
print('file2: \n', file2.info())  # một số NaN trong cột vendor -- một số cv_id không liên kết với bất kỳ vendor nào (cùng số # như bị thiếu đối với product)
print('file3: \n', file3.info())  # ~một nửa số cột đã hoàn thành, ~1/2 còn thiếu một số lượng nhỏ giá trị [số lượng giá trị trong mỗi cột giống nhau]
print('file4: \n', file4.info())  # không có giá trị null'
# Mối quan hệ giữa các bảng trong tập dữ liệu:
# Tệp chính về cves sử dụng có vẻ là tệp 3:
    # file1 cung cấp kết nối giữa cve_id và product
    # file 2 cung cấp kết nối giữa cve_id và vendor
    # file 4 cung cấp kết nối giữa vendor và product
    
# Nhìn vào các giá trị trùng lặp:
lendupfile1 = len(file1[file1['cve_id'].duplicated() == True])
print(lendupfile1, '  ' , len(file1) - lendupfile1)
print('Number of identical, unique cve values shared in file1,file2 and file3 [all = 89660, the length of file3]')

# hơn một nửa số cve_id bị trùng lặp, cho thấy lỗ hổng ảnh hưởng đến nhiều sản phẩm.
print(sum(file1['cve_id'].unique() == file3['Unnamed: 0'].unique()))

# sự tương ứng hoàn hảo giữa cve_id duy nhất trong file1 và cve_id trong file2 và file3

print(sum(file3['Unnamed: 0'].unique() == file2['Unnamed: 0'].unique()))

# nhưng rõ ràng cve_id tương ứng với nhiều nhà cung cấp/sản phẩm cho nhiều cve_id
print('count of vendor unique values in file 2 and 4')
print(len(file2['vendor'].unique()))
print(len(file4['vendor'].unique()))
print('count of product unique values in file 1 and 4')
print(len(file1['vulnerable_product'].unique()))
print(len(file4['product'].unique()))