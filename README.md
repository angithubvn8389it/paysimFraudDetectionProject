# Phát hiện gian lận giao dịch thẻ theo thời gian thực

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-Spark-E25A1C?logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?logo=mongodb&logoColor=white)

## 1. Giới thiệu dự án

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
- Huấn luyện mô hình phát hiện gian lận.
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
|---|---|
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
| Scala | 2.13 |
| TensorFlow | 2.x |

---

# 4. Cài đặt môi trường

## 4.1 Cài đặt Java JDK 11

Apache Spark yêu cầu Java Runtime Environment.

Download:

https://adoptium.net/temurin/releases/?version=11

Sau khi cài đặt, kiểm tra:

```bash
java -version
```

Kết quả mong muốn:

```
openjdk version "11.x.x"
```

Thiết lập biến môi trường:

Windows:

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

https://spark.apache.org/downloads.html


Chọn:

```
Spark release:
4.1.2

Package type:
Pre-built for Apache Hadoop 3.3
```

Download file `.tgz`.

Ví dụ:

```
spark-4.1.2-bin-hadoop3.tgz
```

Giải nén:

Ví dụ:

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

PATH:

```
%SPARK_HOME%\bin
```

Kiểm tra:

```bash
spark-shell
```

Nếu thành công sẽ xuất hiện:

```
Spark session available as 'spark'
```

---

# 6. Cài đặt Hadoop Winutils (Windows)

Nếu chạy Spark trên Windows cần Hadoop binary.

Download:

https://github.com/cdarlint/winutils


Copy:

```
winutils.exe
```

vào:

```
C:\hadoop\bin
```

Thiết lập:

```
HADOOP_HOME=C:\hadoop
```

PATH:

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

https://www.mongodb.com/try/download/community


Chọn:

```
MongoDB Community Server
Windows
MSI Installer
```

---

## 7.2 Khởi động MongoDB

Sau khi cài đặt:

Start MongoDB service:

Windows:

```
Services
    |
    MongoDB Server
    |
    Start
```

Hoặc chạy:

```bash
mongod
```

---

Kiểm tra:

```bash
mongosh
```

Nếu kết nối thành công:

```
test>
```

---

# 8. Import dữ liệu vào MongoDB

Dataset sử dụng:

PaySim Fraud Detection Dataset

Download:

https://www.kaggle.com/datasets/ealaxi/paysim1


Sau khi tải dataset:

Ví dụ:

```
dataset/
    paysim.csv
```

---

Import dữ liệu:

```bash
mongoimport ^
--db fraud_detection ^
--collection transactions ^
--type csv ^
--headerline ^
--file paysim.csv
```

Kiểm tra:

```bash
mongosh

use fraud_detection

db.transactions.findOne()
```

---

# 9. Cài đặt Python Environment

Khuyến nghị sử dụng Virtual Environment.

Tạo môi trường:

```bash
python -m venv venv
```

Kích hoạt:

Windows:

```bash
venv\Scripts\activate
```

---

Cài thư viện:

```bash
pip install -r requirements.txt
```

---

File requirements bao gồm:

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

MongoDB Spark Connector cho phép Spark đọc/ghi trực tiếp dữ liệu MongoDB.

## Phiên bản sử dụng

```
MongoDB Spark Connector:
10.5.x

Spark:
4.1.2

Scala:
2.13
```

---

## Cách 1: Cấu hình khi chạy Spark

Thêm package:

```bash
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0
```

Ví dụ:

```bash
spark-submit ^
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 ^
main.py
```

---

## Cách 2: Cấu hình trong SparkSession

Trong Python:

```python
from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("FraudDetection") \
    .config(
        "spark.mongodb.read.connection.uri",
        "mongodb://localhost:27017/fraud_detection.transactions"
    ) \
    .config(
        "spark.mongodb.write.connection.uri",
        "mongodb://localhost:27017/fraud_detection.results"
    ) \
    .getOrCreate()
```

---

# 11. Chạy chương trình

## 11.1 Clone repository

```bash
git clone <repository-url>

cd BigData-Fraud-Detection
```

---

## 11.2 Kiểm tra MongoDB

Đảm bảo MongoDB đang chạy:

```bash
mongosh
```

---

## 11.3 Chạy chương trình Spark

Ví dụ:

```bash
spark-submit ^
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 ^
src/main.py
```

---

# 12. Cấu trúc thư mục

```
BigData-Fraud-Detection/

│
├── dataset/
│   └── paysim.csv
│
├── src/
│   │
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   └── main.py
│
├── models/
│   ├── random_forest.pkl
│   └── dnn_model.keras
│
├── results/
│   ├── predictions.csv
│   └── evaluation_metrics.txt
│
├── requirements.txt
│
└── README.md
```

---

# 13. Kết quả đầu ra

Sau khi chạy hệ thống:

MongoDB sẽ chứa collection:

```
fraud_detection.results
```

Bao gồm:

| Field | Ý nghĩa |
|-|-|
| transaction_id | Mã giao dịch |
| features | Các thuộc tính giao dịch |
| prediction | Kết quả dự đoán |
| probability | Xác suất gian lận |

Ví dụ:

```json
{
    "transaction_id": 12345,
    "prediction": 1,
    "probability": 0.98
}
```

Trong đó:

```
0 : Giao dịch bình thường

1 : Giao dịch gian lận
```

---

# 14. Các lỗi thường gặp

## Spark không tìm thấy Java

Lỗi:

```
JAVA_HOME is not set
```

Khắc phục:

Kiểm tra:

```bash
echo %JAVA_HOME%
```

---

## MongoDB Connector không tải được

Kiểm tra:

```bash
spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0
```

Đảm bảo có Internet khi chạy lần đầu.

---

## Spark lỗi Hadoop

Lỗi:

```
winutils.exe not found
```

Khắc phục:

Kiểm tra:

```
C:\hadoop\bin\winutils.exe
```

---

# 15. Tác giả

Sinh viên thực hiện:

**Đặng Đức An**

Môn học:

**Nhập môn Phân tích Dữ liệu lớn**

Trường:

**Đại học Văn Lang**

Năm học: 2026
