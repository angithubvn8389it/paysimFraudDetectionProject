# Paysim Fraud Detection Pipeline

This project implements an end-to-end Machine Learning pipeline to detect fraudulent transactions using the Paysim dataset. 

## Hướng dẫn (Tiếng Việt)

Dự án này triển khai một pipeline học máy đầu-cuối để phát hiện giao dịch gian lận trên dữ liệu Paysim.

### Yêu cầu
- Python 3.8+ và `pip`
- Java (bản tương thích với PySpark nếu cần)
- MongoDB (local hoặc Atlas)
- Các thư viện Python trong `requirements.txt`

### Cài đặt nhanh
1. Cài đặt các thư viện:
```powershell
python -m pip install -r requirements.txt
```
2. Cấu hình MongoDB: chỉnh `spark/preprocessing/preprocess.py` nếu bạn dùng MongoDB Atlas hoặc thay đổi URI kết nối.
3. Nếu muốn import dữ liệu raw vào MongoDB (tùy chọn):
```powershell
python mongodb/import_data.py
```

### Chạy pipeline
Chạy toàn bộ pipeline (Spark + huấn luyện RF + lưu kết quả):
```powershell
python main.py
```

Sau khi chạy xong, kết quả và metrics sẽ được lưu vào thư mục `output/`:
- `output/metrics.csv`, `output/metrics.json` — số liệu đánh giá trên tập train/validation/test
- `output/analysis/` — biểu đồ ROC/PR và ma trận nhầm lẫn (nếu bạn chạy script phân tích)
- `models/` — mô hình được lưu (`models/fraud_rf_model` cho Random Forest, `models/baseline_lr` cho baseline Logistic Regression)

### Chạy baseline (Logistic Regression)
Đã có script baseline để so sánh:
```powershell
python baseline.py
```
Metrics baseline sẽ được lưu tại `output/metrics_baseline.csv`.

### Đo thời gian thực thi
Script `timing.py` đo thời gian từng giai đoạn cho cả mô hình baseline và Random Forest (Spark) và lưu vào `output/timing_models.txt`.

### Gợi ý cải tiến tiếp theo
- Điều chỉnh ngưỡng (threshold) để tối ưu Precision/Recall cho lớp gian lận.
- Thử các mô hình khác như XGBoost hoặc LightGBM (trên dữ liệu mẫu hoặc chuyển sang Pandas/Scikit-learn nếu không dùng Spark).
- Thử cân bằng lớp bằng SMOTE hoặc stratified sampling nếu cần.

Nếu bạn muốn, tôi có thể tạo một file `README_vi.md` riêng chứa hướng dẫn này hoặc cập nhật nội dung chi tiết hơn.

## Architecture & Technologies
- **PySpark**: For distributed data processing, feature engineering, and training the Random Forest classification model.
- **MongoDB**: Used as the primary data store for the raw transactions and the final model predictions.
- **Pandas / Imbalanced-learn**: Supported for SMOTE in partitioned processing, though native PySpark undersampling is the default approach for local stability.

## Project Structure
- `main.py`: The entry point script that orchestrates the whole ML pipeline.
- `mongodb/`: Scripts related to importing raw CSV data into MongoDB.
- `spark/`: 
  - `preprocessing/`: Scripts for data loading, cleaning, categorical encoding, and dealing with class imbalances.
  - `training/`: Model training logic (Random Forest Classifier) using cross-validation.
  - `evaluation/`: Scripts to evaluate the model on metrics like AUROC, AUPR, F1-Score, and accuracy.
- `visualization/`: Scripts to generate visual reports and dashboards based on predictions.
- `models/`: The saved PySpark ML models are exported here.

## How to Run

1. **Setup MongoDB**: 
   Ensure you have access to a MongoDB instance (local or Atlas) and update the connection URI in `spark/preprocessing/preprocess.py`. 
   If using local MongoDB, you can run `mongodb/import_data.py` to seed the database from your raw Paysim CSV file.
   
2. **Install Dependencies**: 
   Install required packages via `pip install -r requirements.txt`. (Make sure your PySpark version is compatible with the MongoDB spark connector).

3. **Execute Pipeline**: 
   Run the main orchestrator script:
   ```bash
   python main.py
   ```
   This will initialize Spark, load data from MongoDB, preprocess it, train the model, evaluate the metrics, generate visualizations, and write predictions back to MongoDB.

## Pipeline Steps
1. **Initialization**: Configures Spark with the necessary MongoDB connectors and memory settings.
2. **Data Loading**: Reads transactions directly from MongoDB, utilizing pushdown filters for memory efficiency.
3. **Preprocessing**: Cleans the data, drops unnecessary high-cardinality columns, encodes categoricals manually to avoid eager evaluation overhead, and casts numeric types.
4. **Resampling**: Due to the heavy class imbalance in fraud data, the majority class is downsampled to match the minority class before training.
5. **Model Training**: A Random Forest model is trained with 3-fold Cross Validation to tune hyperparameters (`maxDepth` and `numTrees`).
6. **Evaluation & Output**: The model is evaluated against a hold-out test set, and the predictions (including predicted probabilities) are saved back to a new MongoDB collection (`fraudDetection.fraudResults`).
