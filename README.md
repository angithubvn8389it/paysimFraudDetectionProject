# Phát hiện gian lận theo thời gian thực theo bộ dữ liệu Paysim

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-Spark-E25A1C?logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?logo=mongodb&logoColor=white)

Dự án này xây dựng một pipeline học máy đầu-cuối để phát hiện giao dịch gian lận trên bộ dữ liệu Paysim. Hệ thống sử dụng PySpark để xử lý dữ liệu lớn, huấn luyện mô hình Random Forest, đánh giá kết quả và xuất các biểu đồ, metrics cũng như dự đoán ra file hoặc MongoDB.

## Giới thiệu

Mục tiêu của dự án là:

- phát hiện các giao dịch có khả năng gian lận;
- xử lý dữ liệu giao dịch lớn bằng Spark;
- so sánh nhanh giữa mô hình baseline và mô hình Random Forest;
- lưu lại kết quả đánh giá để dễ theo dõi và tái sử dụng.

## Cách nhập dự án vào máy

Nếu bạn muốn đưa repo này về máy để chạy hoặc chỉnh sửa, hãy làm theo các bước sau:

1. Mở terminal hoặc PowerShell tại thư mục bạn muốn lưu dự án.
2. Clone repo từ GitHub:

```powershell
git clone https://github.com/angithubvn8389it/paysimFraudDetectionProject.git
```

3. Di chuyển vào thư mục dự án:

```powershell
cd paysimFraudDetectionProject
```

4. Mở dự án trong VS Code nếu cần:

```powershell
code .
```

Nếu bạn tải source bằng file `.zip` thay vì clone, chỉ cần giải nén vào một thư mục và mở thư mục đó trong VS Code.

## Thông tin về chương trình

Các thành phần chính trong dự án:

- `main.py`: chạy toàn bộ pipeline huấn luyện và đánh giá.
- `baseline.py`: chạy mô hình baseline để so sánh.
- `timing.py`: đo thời gian thực thi các giai đoạn của pipeline.
- `visualization/visualize.py`: tạo các biểu đồ đánh giá như ma trận nhầm lẫn, ROC, Precision-Recall và feature importance.
- `mongodb/import_data.py`: nhập dữ liệu thô vào MongoDB.
- `spark/`: chứa các bước tiền xử lý, huấn luyện và đánh giá trên Spark.

Thư mục đầu ra thường gồm:

- `output/metrics.csv`, `output/metrics.json`: các chỉ số đánh giá của mô hình.
- `output/metrics_baseline.csv`: kết quả của mô hình baseline.
- `output/timing_models.txt`: thời gian chạy từng giai đoạn của pipeline.
- `output/`: các biểu đồ và báo cáo được tạo ra trong quá trình chạy.

## Yêu cầu cài đặt

Trước khi chạy dự án, bạn cần có:

- Python 3.8 trở lên;
- Java tương thích với PySpark;
- MongoDB local hoặc MongoDB Atlas;
- các thư viện Python trong `requirements.txt`.

## Cài đặt

1. Cài các thư viện cần thiết:

```powershell
python -m pip install -r requirements.txt
```

2. Cấu hình MongoDB nếu cần:

- kiểm tra và chỉnh URI kết nối trong `spark/preprocessing/preprocess.py`;
- nếu dùng MongoDB local, bạn có thể import dữ liệu thô trước khi chạy pipeline.

3. Nạp dữ liệu vào MongoDB (tùy chọn):

```powershell
python mongodb/import_data.py
```

## Cách sử dụng

### Chạy pipeline chính

Lệnh sau sẽ khởi chạy toàn bộ quy trình: đọc dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá và lưu kết quả.

```powershell
python main.py
```

### Chạy mô hình baseline

Nếu bạn muốn chạy mô hình baseline để so sánh với Random Forest:

```powershell
python baseline.py
```

### Đo thời gian thực thi

Để ghi lại thời gian cho từng giai đoạn của pipeline:

```powershell
python timing.py
```

## Kết quả sau khi chạy

Sau khi chạy xong, bạn sẽ thấy các kết quả chính được lưu trong thư mục `output/` và `models/`:

- các file metrics để theo dõi hiệu năng mô hình;
- các biểu đồ đánh giá như ROC, Precision-Recall, ma trận nhầm lẫn;
- mô hình đã huấn luyện được lưu trong `models/`;
- dự đoán cuối cùng có thể được ghi vào MongoDB.

## Cấu trúc thư mục

- `data/`: dữ liệu raw và dữ liệu đã xử lý.
- `models/`: mô hình đã huấn luyện.
- `mongodb/`: script nhập dữ liệu và tạo index MongoDB.
- `output/`: metrics, biểu đồ và các file kết quả.
- `spark/`: mã nguồn cho xử lý dữ liệu, huấn luyện và đánh giá bằng Spark.
- `visualization/`: các script trực quan hóa.
