# Phát hiện gian lận giao dịch thẻ theo thời gian thực

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-Spark%204.1.2-E25A1C?logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-8.x-47A248?logo=mongodb&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)

# 1. Giới thiệu dự án

Dự án **"Phát hiện gian lận giao dịch thẻ theo thời gian thực"** xây dựng hệ thống phát hiện các giao dịch gian lận dựa trên công nghệ Big Data kết hợp với Machine Learning và Deep Learning.

Hệ thống sử dụng:

- **MongoDB**: Lưu trữ dữ liệu giao dịch dưới dạng NoSQL.
- **Apache Spark / PySpark**: Xử lý dữ liệu phân tán.
- **Spark MLlib**: Huấn luyện mô hình Machine Learning.
- **TensorFlow/Keras**: Xây dựng mô hình Deep Neural Network.
- **MongoDB Spark Connector**: Kết nối trực tiếp giữa MongoDB và Spark.

Mục tiêu của hệ thống:

- Thu thập và lưu trữ dữ liệu giao dịch lớn.
- Tiền xử lý dữ liệu bằng Spark.
- Xây dựng mô hình phát hiện gian lận.
- Dự đoán giao dịch bất thường.
- Lưu kết quả dự đoán trở lại MongoDB.

---

# 2. Kiến trúc hệ thống

Luồng xử lý chính:

```
Transaction Dataset
        |
        v
     MongoDB
        |
        v
 MongoDB Spark Connector
        |
        v
     Apache Spark
        |
        v
 Data Preprocessing
        |
        v
 Feature Engineering
        |
        v
 Machine Learning / Deep Learning Model
        |
        v
 Fraud Prediction
        |
        v
 Save Result to MongoDB
```

---

# 3. Yêu cầu hệ thống

## Hardware tối thiểu

| Thành phần | Yêu cầu |
|-|-|
| CPU | Intel Core i5 hoặc tương đương |
| RAM | 8GB trở lên |
| Storage | Tối thiểu 10GB trống |

---

## Software requirements

| Phần mềm | Phiên bản đề xuất |
|-|-|
| Python | 3.11 |
| Java JDK | 11 |
| Apache Spark | 4.1.2 |
| Hadoop | 3.3.4 |
| MongoDB | 8.x |
| MongoDB Compass | Latest |
| Scala | 2.13 |
| TensorFlow | 2.x |

---

# 4. Cài đặt môi trường

## 4.1 Cài đặt Java JDK 11

Apache Spark yêu cầu Java Runtime Environment.

Download:

```
https://adoptium.net/temurin/releases/?version=11
```

Sau khi cài đặt, kiểm tra:

```bash
java -version
```

Kết quả mong muốn:

```
openjdk version "11.x.x"
```

Thiết lập biến môi trường Windows:

```
JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.x.x
```

Thêm vào PATH:

```
%JAVA_HOME%\bin
```

---

# 5. Cài đặt Apache Spark

## 5.1 Download Spark

Truy cập:

```
https://spark.apache.org/downloads.html
```

Chọn:

```
Spark release:
4.1.2

Package type:
Pre-built for Apache Hadoop 3.3
```

Download file:

```
spark-4.1.2-bin-hadoop3.tgz
```

Giải nén, ví dụ:

```
C:\spark
```

---

## 5.2 Thiết lập biến môi trường Spark

Windows Environment Variables:

Thêm:

```
SPARK_HOME=C:\spark
```

Thêm vào PATH:

```
%SPARK_HOME%\bin
```

Kiểm tra:

```bash
spark-shell
```

Nếu cài đặt thành công:

```
Spark session available as 'spark'
```

---

# 6. Cài đặt Hadoop Winutils (Windows)

Khi chạy Spark trên Windows cần Hadoop binary để xử lý file system.

Download:

```
https://github.com/cdarlint/winutils
```

Copy file:

```
winutils.exe
```

vào:

```
C:\hadoop\bin
```

Thiết lập biến môi trường:

```
HADOOP_HOME=C:\hadoop
```

Thêm PATH:

```
%HADOOP_HOME%\bin
```

Kiểm tra:

```bash
winutils.exe ls
```

---

# 7. Cài đặt MongoDB

## 7.1 Download MongoDB Community Server

Download:

```
https://www.mongodb.com/try/download/community
```

Chọn:

```
MongoDB Community Server

Windows

MSI Installer
```

---

## 7.2 Cài đặt MongoDB Server

Trong quá trình cài đặt:

Chọn:

```
Complete Installation
```

Nên chọn:

```
Install MongoDB as a Service
```

để MongoDB tự động chạy cùng Windows.

Sau khi hoàn tất, kiểm tra:

```bash
mongosh
```

Nếu kết nối thành công:

```
test>
```

MongoDB Server đã hoạt động.

---

# 7.3 Cài đặt MongoDB Compass

MongoDB Compass là công cụ giao diện đồ họa (GUI) chính thức của MongoDB, giúp quản lý Database, Collection, xem dữ liệu và kiểm tra kết quả dự đoán.

Trong dự án này, MongoDB Compass được sử dụng để:

- Kiểm tra dữ liệu sau khi import.
- Quan sát collection `paysimData`.
- Theo dõi kết quả dự đoán trong collection `fraudResults`.
- Thực hiện truy vấn dữ liệu.

---

## 7.3.1 Download MongoDB Compass

Truy cập:

```
https://www.mongodb.com/products/tools/compass
```

Chọn:

```
MongoDB Compass

Windows

Download
```

File tải về:

```
mongodb-compass-*.msi
```

---

## 7.3.2 Cài đặt MongoDB Compass

Mở file:

```
mongodb-compass-*.msi
```

Thực hiện:

```
Next
    |
Accept License Agreement
    |
Complete Installation
    |
Install
```

Sau khi cài đặt hoàn tất, mở:

```
Start Menu
    |
    └── MongoDB Compass
```

---

## 7.3.3 Kết nối MongoDB bằng Compass

Đảm bảo MongoDB Server đang chạy.

Kiểm tra:

```bash
mongosh
```

Mở MongoDB Compass.

Tại màn hình:

```
New Connection
```

Nhập:

```
mongodb://localhost:27017
```

Chọn:

```
Connect
```

---

Nếu kết nối thành công, MongoDB Compass sẽ hiển thị:

```
Databases

├── admin
├── config
├── local
└── fraudDetection
```

---

## 7.3.4 Kiểm tra Database và Collection

Database sử dụng trong dự án:

```
fraudDetection
```

Các collection:

```
fraudDetection

├── paysimData
│
└── fraudResults
```

Trong đó:

| Collection | Chức năng |
|-|-|
| paysimData | Dữ liệu giao dịch ban đầu từ Paysim |
| fraudResults | Kết quả dự đoán gian lận |

---

## 7.3.5 Xem dữ liệu bằng MongoDB Compass

Chọn:

```
fraudDetection
        |
        └── fraudResults
```

Chọn tab:

```
Documents
```

để xem các kết quả dự đoán được Spark ghi vào MongoDB.
---

# 8. Import dữ liệu vào MongoDB

## Dataset sử dụng

Dataset:

```
PaySim Fraud Detection Dataset
```

Nguồn tải:

```
https://www.kaggle.com/datasets/ealaxi/paysim1
```

Sau khi tải dataset:

Ví dụ cấu trúc:

```
dataset/

└── paysimLog.csv
```

---

## Import dữ liệu bằng mongoimport

Mở Command Prompt tại thư mục chứa file dataset.

Chạy:

```bash
mongoimport ^
--db fraudDetection ^
--collection paysimData ^
--type csv ^
--headerline ^
--file paysimLog.csv
```

Giải thích:

| Tham số | Ý nghĩa |
|-|-|
| `--db` | Database MongoDB |
| `--collection` | Collection lưu dữ liệu |
| `--type csv` | File dữ liệu dạng CSV |
| `--headerline` | Sử dụng dòng đầu làm tên cột |
| `--file` | Đường dẫn file dữ liệu |

---

## Kiểm tra dữ liệu sau khi import

Mở MongoDB Shell:

```bash
mongosh
```

Chọn database:

```javascript
use fraudDetection
```

Kiểm tra collection:

```javascript
show collections
```

Kết quả:

```
paysimData
```

Xem một document:

```javascript
db.transactions.findOne()
```

---

# 9. Cài đặt Python Environment

Khuyến nghị sử dụng Virtual Environment để quản lý thư viện.

## 9.1 Tạo môi trường ảo

```bash
python -m venv venv
```

---

## 9.2 Kích hoạt môi trường

Windows:

```bash
venv\Scripts\activate
```

Sau khi kích hoạt thành công:

```
(venv)
```

sẽ xuất hiện ở đầu dòng lệnh.

---

## 9.3 Cài đặt thư viện

Chạy:

```bash
pip install -r requirements.txt
```

---

## Các thư viện chính

```
pyspark
pymongo
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
```

---

# 10. Cài đặt MongoDB Spark Connector

MongoDB Spark Connector cho phép Apache Spark đọc và ghi dữ liệu trực tiếp với MongoDB.

## Phiên bản sử dụng

```
MongoDB Spark Connector:
10.5.0

Apache Spark:
4.1.2

Scala:
2.13
```

---

# 10.1 Cấu hình Connector khi chạy Spark

Thêm package:

```bash
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0
```

Ví dụ:

```bash
spark-submit ^
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 ^
src/main.py
```

Spark sẽ tự động tải MongoDB Connector trong lần chạy đầu tiên.

---

# 10.2 Cấu hình trong SparkSession

Ví dụ:

```python
from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("FraudDetection") \
    .config(
        "spark.mongodb.read.connection.uri",
        "mongodb://localhost:27017/fraudDetection.paysimData"
    ) \
    .config(
        "spark.mongodb.write.connection.uri",
        "mongodb://localhost:27017/fraudDetection.fraudResults"
    ) \
    .getOrCreate()
```

---

# 11. Chạy chương trình

## 11.1 Clone repository

```bash
git clone <repository-url>

cd paysimFraudDetectionProject
```

---

## 11.2 Kiểm tra MongoDB

Đảm bảo MongoDB Server đang chạy:

```bash
mongosh
```

Kiểm tra:

```javascript
show databases
```

Database:

```
fraudDetection
```

---

## 11.3 Chạy chương trình bằng Spark

Ví dụ:

```bash
spark-submit ^
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 ^
src/main.py
```

---

Sau khi chạy hoàn tất:

Spark sẽ:

1. Đọc dữ liệu từ MongoDB collection:

```
fraudDetection.paysimData
```

2. Tiền xử lý dữ liệu.

3. Chạy mô hình Machine Learning / Deep Learning.

4. Ghi kết quả dự đoán vào:

```
fraudDetection.fraudResults
```

---

# 12. Cấu trúc thư mục

```
paysimFraudDetectionProject/
├── baseline.py
├── main.py
├── timing.py
├── README.md
├── data/
│   ├── raw_data/
│   └── processed/
├── models/
│   ├── baseline_lr/
│   └── fraud_rf_model/
├── mongodb/
│   ├── create_indexes.js
│   └── import_data.py
├── output/
│   ├── analysis/
│   ├── eda/
│   ├── metrics.csv
│   ├── metrics.json
│   ├── metrics.txt
│   ├── metrics_baseline.csv
│   ├── metrics_baseline.txt
│   ├── timing_models.txt
│   └── timing_models.png
├── spark/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── streaming/
│   └── training/
└── visualization/
    └── visualize.py
```
---

# 13. Kết quả đầu ra

Sau khi hệ thống hoàn thành dự đoán, kết quả được lưu vào MongoDB:

Database:

```
fraudDetection
```

Collection:

```
fraudResults
```

---

## Cấu trúc Document kết quả

Ví dụ:

```json
{
  "_id": {
    "$oid": "6a6178775d3447549ed31fe8"
  },
  "step": 537,
  "typeIndex": 0,
  "amount": 1504.73,
  "oldbalanceOrg": 1504.73,
  "newbalanceOrig": 0,
  "oldbalanceDest": 26284.34,
  "newbalanceDest": 27789.07,
  "actual_isFraud": 1,
  "predicted_isFraud": 1,
  "probability": [
    0.4729327961625122,
    0.5270672038374878
  ]
}
```

---

## Ý nghĩa các trường dữ liệu

| Trường | Ý nghĩa |
|-|-|
| `_id` | ID tự động của MongoDB |
| `step` | Bước thời gian của giao dịch |
| `typeIndex` | Loại giao dịch sau khi mã hóa |
| `amount` | Số tiền giao dịch |
| `oldbalanceOrg` | Số dư tài khoản gửi trước giao dịch |
| `newbalanceOrig` | Số dư tài khoản gửi sau giao dịch |
| `oldbalanceDest` | Số dư tài khoản nhận trước giao dịch |
| `newbalanceDest` | Số dư tài khoản nhận sau giao dịch |
| `actual_isFraud` | Nhãn thực tế của giao dịch |
| `predicted_isFraud` | Nhãn dự đoán của mô hình |
| `probability` | Xác suất dự đoán của mô hình |

---

# 13.1 Ý nghĩa kết quả dự đoán

## actual_isFraud

Nhãn thực tế:

```
0: Giao dịch bình thường

1: Giao dịch gian lận
```

---

## predicted_isFraud

Kết quả mô hình:

```
0: Dự đoán giao dịch bình thường

1: Dự đoán giao dịch gian lận
```

---

## probability

Ví dụ:

```json
[
0.4729327961625122,
0.5270672038374878
]
```

Ý nghĩa:

| Vị trí | Ý nghĩa |
|-|-|
| probability[0] | Xác suất giao dịch bình thường |
| probability[1] | Xác suất giao dịch gian lận |

Kết quả:

```
Normal:
47.29%

Fraud:
52.71%
```

Vì:

```
probability[1] > probability[0]
```

nên:

```
predicted_isFraud = 1
```

---

# 13.2 Kiểm tra kết quả bằng MongoDB Compass

## Tìm các giao dịch được dự đoán gian lận

Trong ô Filter:

```json
{
  "predicted_isFraud": 1
}
```

---

## Tìm các giao dịch dự đoán đúng

```json
{
  "actual_isFraud": 1,
  "predicted_isFraud": 1
}
```

Đây là:

```
True Positive
```

---

## Tìm các giao dịch bị bỏ sót

```json
{
  "actual_isFraud": 1,
  "predicted_isFraud": 0
}
```

Đây là:

```
False Negative
```

---

# 14. Các lỗi thường gặp

## 14.1 Spark không tìm thấy Java

Lỗi:

```
JAVA_HOME is not set
```

Khắc phục:

Kiểm tra:

```bash
echo %JAVA_HOME%
```

Đảm bảo:

```
JAVA_HOME
```

trỏ đúng tới thư mục JDK 11.

---

## 14.2 MongoDB Connector không tải được

Kiểm tra:

```bash
spark-submit ^
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0
```

Lưu ý:

- Cần Internet trong lần chạy đầu tiên.
- Kiểm tra đúng phiên bản Scala.

---

## 14.3 Spark lỗi Hadoop

Lỗi:

```
winutils.exe not found
```

Khắc phục:

Kiểm tra:

```
C:\hadoop\bin\winutils.exe
```

và biến môi trường:

```
HADOOP_HOME=C:\hadoop
```

---

## 14.4 MongoDB Compass không thấy dữ liệu

Kiểm tra:

1. MongoDB Server đang chạy.

2. Đúng Database:

```
fraud_detection
```

3. Đúng Collection:

```
fraudResults
```

4. Nhấn:

```
Refresh
```

trong MongoDB Compass.

---

# 15. Tác giả

Sinh viên thực hiện: **Đặng Đức An**

Môn học: **Nhập môn Phân tích Dữ liệu lớn**

Trường: **Trường Đại học Văn Lang**

Năm học: **2025-2026**
