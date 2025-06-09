# Thuật toán định lượng rủi ro bảo mật trong DevOps

Một đồ án thử nghiệm, xây dựng các thuật toán, mô hình phân loại tấn công bảo mật, rủi ro bảo mật được xây dựng bằng Python, triển khai qua Docker + FasAPI và được xây dựng FE hoàn thiện mô phỏng ứng dụng thực tế

![GitHub issues](https://img.shields.io/github/issues/nguyendang0106/security-risk-classification-algorithm)

## 📖 Giới thiệu

Đây là đồ án Project 2 của tôi. Đồ án giải quyết vấn đề phân loại tấn công bảo mật bằng cách cung cấp các chức năng chính sau:
* Chức năng 1: Xây dựng mô hình, kiến trúc cốt lõi ML phân loại tấn công.
* Chức năng 2: Triển khai API.
* Chức năng 3: Xây dựng FE.


## 🛠️ Công nghệ sử dụng

* **Frontend:** HTML/CSS/JS
* **Backend:** Docker, FastAPI

## ⚙️ Hướng dẫn cài đặt

Thực hiện các bước sau để chạy dự án trên máy của bạn:

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/nguyendang0106/security-risk-classification-algorithm.git](https://github.com/nguyendang0106/security-risk-classification-algorithm.git)
    ```
2.  **Di chuyển vào thư mục dự án:**
    ```bash
    cd security-risk-classification-algorithm
    ```
3.  **Cài đặt các thư viện:**
    ```bash
    pip install requirements.txt
    ```
4.  **Khởi chạy dự án:**
    * Build Docker: `docker build -t prj2_20242 .` 
    * Chạy Docker: `docker run -p 8888:8888 prj2_20242`

##  Hướng dẫn sử dụng

Sau khi khởi chạy thành công, bạn có thể:
1.  Mở trình duyệt và truy cập vào `http://0.0.0.0:8888/docs`
2.  Thử nghiệm các API.
3.  Sử dụng các chức năng chính của FE bằng cách nhấn vào index.html trong thư mục fe2.

## Tiến độ thực hiện theo tuần

Dưới đây là nhật ký ghi lại quá trình thực hiện đồ án.

### Tuần 1: Nhận Công việc, Viết Đề cương, Lập kế hoạch, Chuẩn bị các yêu cầu đề thực hiện công việc, Nộp Đề cương (Có xác nhận của Thầy hướng dẫn và Bộ môn)
### Tuần 2: Bộ dữ liệu đã qua tiền xử lý.
### Tuần 3: Mô hình đầu tiên hoạt động, có thể so sánh.
### Tuần 4: Mô hình chính xác hơn baseline.
### Tuần 5: Mô hình tối ưu, sẵn sàng triển khai vào API.
### Tuần 6: API có thể nhận request và trả kết quả từ mô hình.
### Tuần 7: API hoạt động ổn định, có test cơ bản.
### Tuần 8: API có thể chạy trong Docker container.
### Tuần 9: API v2 với mô hình tối ưu hơn.
### Tuần 10: API ổn định, đã kiểm thử kỹ.
### Tuần 11: API có thể truy cập từ internet.
### Tuần 12: API chịu tải tốt, có thể sử dụng thực tế.
### Tuần 13: Giao diện cơ bản hoạt động.
### Tuần 14: UI có thể gửi request đến API và nhận phản hồi.
### Tuần 15: Tài liệu mô tả mô hình & API.
### Tuần 16: Dự án hoàn chỉnh, sẵn sàng báo cáo.

## Tác giả

* **Nguyễn Tiến Đăng:** [Nguyễn Tiến Đăng]
* **Email:** [nguyentiendang0106@gmail.com]
* **GitHub:** [github.com/nguyendang0106](https://github.com/nguyendang0106)
