# Thuật toán định lượng rủi ro bảo mật trong DevOps

Một đồ án thử nghiệm, xây dựng các thuật toán, mô hình phân loại tấn công bảo mật, rủi ro bảo mật được xây dựng bằng Python, triển khai qua Docker + FasAPI và được xây dựng FE hoàn thiện mô phỏng ứng dụng thực tế

![GitHub issues](https://img.shields.io/github/issues/nguyendang0106/security-risk-classification-algorithm)

## Giới thiệu

Đây là đồ án Project 2 của tôi. Đồ án giải quyết vấn đề phân loại tấn công bảo mật bằng cách cung cấp các chức năng chính sau:
* Chức năng 1: Xây dựng mô hình, kiến trúc cốt lõi ML phân loại tấn công.
* Chức năng 2: Triển khai API.
* Chức năng 3: Xây dựng FE.


## Công nghệ sử dụng

* **Frontend:** HTML/CSS/JS
* **Backend:** Docker, FastAPI

## Hướng dẫn cài đặt

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

## Hướng dẫn sử dụng

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

## Bối cảnh sử dụng

### Kịch bản 1: Phân tích Nhanh & Điều tra Sự cố Đơn lẻ (Sử dụng chức năng "Nhập dữ liệu trực tiếp")

Bối cảnh:
Một chuyên viên phân tích A đang theo dõi hệ thống cảnh báo an ninh (SIEM). Bất ngờ, hệ thống SIEM (Security Information and Event Management) gửi một cảnh báo về một kết nối bất thường từ máy tính của nhân viên kế toán (PC-KT01) đến một địa chỉ IP lạ trên internet qua một cổng không phổ biến (ví dụ: port 6667). Cảnh báo này chỉ là mức độ thấp vì nó chỉ dựa trên quy tắc đơn giản (kết nối tới IP/port lạ).

Hành động:
Thu thập đặc trưng: A cần một "ý kiến thứ hai" nhanh chóng và thông minh hơn. Anh ấy sử dụng các công cụ có sẵn (như Wireshark hoặc trích xuất từ log của Firewall/Netflow) để lấy các đặc trưng chi tiết của luồng kết nối đó: Flow Duration, Total Fwd Packets, Fwd Packet Length Max, Bwd Packet Length Mean, Protocol...
Sử dụng phần mềm: A mở ứng dụng, chọn chức năng "Nhập dữ liệu trực tiếp". Anh ấy điền các giá trị đặc trưng vừa thu thập vào các ô tương ứng.
Nhấn "Phân loại".

Kết quả & Giá trị:
Chương trình trả về kết quả: Botnet Attack.
Giá trị thực tế: Thay vì một cảnh báo chung chung, A giờ đã có một nhận định cụ thể và đáng tin cậy hơn. Anh ấy có thể ngay lập tức chuyển hướng điều tra: thay vì chỉ kiểm tra tường lửa, anh sẽ tập trung vào việc cô lập máy tính PC-KT01 và tiến hành quét tìm phần mềm độc hại (malware), kiểm tra các tiến trình lạ, và truy tìm các dấu hiệu của botnet. Điều này giúp rút ngắn đáng kể thời gian phản ứng và ngăn chặn mối đe dọa trước khi nó lan rộng.

### Kịch bản 2: Phân tích Log Hàng loạt & Truy vết Sau Sự cố (Sử dụng chức năng "Tải lên tệp CSV")

Kịch bản này dành cho đội phản ứng sự cố (Incident Response Team) hoặc các chuyên viên phân tích khi cần điều tra sâu một vụ việc đã xảy ra.
Bối cảnh:
Công ty phát hiện một trong những máy chủ web (Web Server 01) đã bị xâm nhập và sử dụng để tấn công từ chối dịch vụ (DDoS) vào một mục tiêu khác. Vụ việc được cho là đã diễn ra vào ngày hôm qua. Đội phản ứng sự cố cần phân tích toàn bộ lưu lượng mạng ra vào máy chủ đó trong 24 giờ qua để hiểu rõ chuỗi tấn công.

Hành động:
Trích xuất dữ liệu: Quản trị viên mạng trích xuất toàn bộ bản ghi lưu lượng (netflow/sflow) liên quan đến địa chỉ IP của Web Server 01 trong khoảng thời gian nghi vấn. Dữ liệu này được xử lý và xuất ra thành một tệp webserver01_traffic_log.csv với hàng chục nghìn dòng, mỗi dòng là một luồng kết nối với đầy đủ các cột đặc trưng.
Sử dụng phần mềm: Một thành viên trong đội phản ứng sự cố mở ứng dụng, chọn chức năng "Tải lên tệp CSV" và tải lên tệp webserver01_traffic_log.csv.
Chờ xử lý.

Kết quả & Giá trị:
Chương trình trả về một danh sách tương ứng với loại tấn công cho mỗi dữ liệu.
Giá trị thực tế: Đội điều tra có thể lọc và sắp xếp tệp kết quả này. Họ nhanh chóng phát hiện ra:
Vài dòng được phân loại là Web Attack - Brute Force: Dấu hiệu kẻ tấn công dò mật khẩu quản trị.
Một dòng được phân loại là Infiltration: Đây có thể là thời điểm kẻ tấn công đã vào được bên trong.
Hàng nghìn dòng sau đó được phân loại là DDoS: Dấu hiệu máy chủ đã bị chiếm quyền và trở thành một phần của mạng botnet.
Nhờ đó, đội có thể tái hiện lại toàn bộ chuỗi tấn công, xác định được thời điểm, phương thức xâm nhập và quy mô của thiệt hại một cách hiệu quả thay vì phải đọc thủ công hàng nghìn dòng log vô nghĩa.

### Kịch bản 3: Diễn tập, Đào tạo & Trình diễn Công nghệ (Sử dụng chức năng "Giám sát hệ thống")

Chức năng này phù hợp cho mục đích đào tạo, trình diễn cho khách hàng, hoặc diễn tập an ninh mạng mô phỏng một cuộc tấn công theo thời gian thực.
Bối cảnh:
Bản thân đang trình diễn giải pháp của mình cho một khách hàng tiềm năng. Hoặc, một giảng viên an ninh mạng muốn cho sinh viên thấy cách một hệ thống phát hiện xâm nhập (IDS) dựa trên AI/ML hoạt động trong thực tế.

Hành động:
Chuẩn bị dữ liệu: Chuẩn bị một tệp live_attack_simulation.csv được sắp xếp theo thứ tự thời gian. Tệp này chứa:
Khoảng 100 dòng đầu là lưu lượng bình thường (Benign).
Tiếp theo là 20 dòng mô phỏng hành vi quét cổng (PortScan).
Tiếp theo là 10 dòng tấn công dò mật khẩu (FTP-Patator).
Cuối cùng là một vài dòng lưu lượng Botnet.
Sử dụng phần mềm:
Mở ứng dụng, chọn chức năng "Giám sát hệ thống".
Tải lên tệp live_attack_simulation.csv làm nguồn dữ liệu.
Nhấn nút "Bắt đầu Giám sát". Màn hình ứng dụng được chiếu lên cho mọi người cùng xem.

Kết quả & Giá trị:
Mọi người sẽ thấy trên màn hình:
Từng dòng dữ liệu được đọc và phân loại là Benign.
Bất ngờ, một cảnh báo hiện lên: Phát hiện PortScan.
Đồng thời, danh sách các cuộc tấn công đã phát hiện được cập nhật liên tục ở một góc màn hình, cho thấy một bức tranh toàn cảnh.
Giá trị thực tế: Kịch bản này tạo ra một trải nghiệm trực quan và sinh động. Nó giúp người xem (khách hàng, sinh viên) hiểu ngay lập tức cách hệ thống hoạt động và giá trị của việc phát hiện tấn công theo thời gian thực. Nó thuyết phục hơn nhiều so với việc chỉ trình bày các slide báo cáo tĩnh. Đối với các đội SOC, họ có thể dùng chức năng này để "tái hiện" lại một cuộc tấn công trong quá khứ và xem mô hình mới sẽ phản ứng ra sao.



## Tác giả

* **Nguyễn Tiến Đăng:**
* **Email:** [nguyentiendang0106@gmail.com]
* **GitHub:** [github.com/nguyendang0106](https://github.com/nguyendang0106)
