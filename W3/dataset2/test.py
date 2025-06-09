import pandas as pd


#access files
file1 = pd.read_csv('W3/dataset2/products.csv')
file2 = pd.read_csv('W3/dataset2/vendors.csv')
file3 = pd.read_csv('W3/dataset2/cve.csv')
file4 = pd.read_csv('W3/dataset2/vendor_product.csv')

# xóa các giá trị null trong file3 vì đối với mỗi cột chúng chiếm <5% giá trị trong các cột đó
file3 = file3.dropna()


## Merge files:
# Có thể làm việc với tệp gốc (tệp 3 hoặc tệp 2/3 được hợp nhất) chỉ để sử dụng ít phép tính hơn cho mỗi phép tính.
file3 = file3.rename(columns = {'Unnamed: 0':'cve_id'})
file2 = file2.rename(columns = {'Unnamed: 0':'cve_id'})

#nếu chúng ta muốn hợp nhất mọi thứ khả năng tệp 4 có thể bỏ qua
file13 = file1.merge(file3)
file123 = file13.merge(file2)

#file123.info()
#file123.head()

sum(file123.duplicated())  

# Chuyển đổi cột ngày thành datetime
file123['mod_date'] = pd.to_datetime(file123['mod_date'])
file123['pub_date'] = pd.to_datetime(file123['pub_date'])

# chỉ cần làm cho ngày mod là một chức năng của ngày xuất bản
file123['mod_date'] = file123['mod_date'] - file123['pub_date'] 

# tất cả các sản phẩm của NaN cũng là nhà cung cấp của NaN
# tóm tắt
pd.options.display.max_colwidth=500
# print(file123[file123['vendor'].isna()]['summary'].head(n=1)) 
# đã kiểm tra một trong các bản tóm tắt, từ chỉ định duy nhất của nhà cung cấp hoặc sản phẩm là 'oVirt'
file123 = file123.dropna()
file123[file123['vulnerable_product'].str.contains('oVirt')]

# không có oVirt trong danh sách sản phẩm -- sẽ chỉ đơn giản là xóa NaN, như đã thực hiện và không cố gắng cứu vãn NaN trong các cột nhà cung cấp/sản phẩm
print(file123.info())

file123.to_csv('W3/dataset2/merged.csv', index=False)