from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv("W2/preprocessed_cve_data.csv")

text = " ".join(df["Description"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Từ phổ biến trong mô tả CVE")
plt.show()