import streamlit as st
import pandas as pd
import pymongo
import os
import json
from PIL import Image

# For Inference
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# --- PAGE CONFIG ---
st.set_page_config(page_title="Phát hiện Gian lận Paysim", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0B0;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES ---
@st.cache_resource
def init_connection():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except Exception as e:
        return None

@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("Paysim Fraud Inference") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    return spark

@st.cache_resource
def load_rf_model(_spark):
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'fraud_rf_model')
    if os.path.exists(model_dir):
        return RandomForestClassificationModel.load(model_dir)
    return None

def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'output', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

client = init_connection()
spark = init_spark()
model = load_rf_model(spark)
metrics = load_metrics()

# --- HEADER ---
st.title("🛡️ Hệ thống Phát hiện Gian lận")
st.markdown("Theo dõi các số liệu phát hiện gian lận theo thời gian thực và kiểm tra các giao dịch.")
st.divider()

# --- TABS ---
tab_overview, tab_inference = st.tabs(["📊 Tổng quan Bảng điều khiển", "🔍 Dự đoán Thời gian Thực"])

with tab_overview:
    if not client:
        st.error("Không thể kết nối đến MongoDB. Vui lòng đảm bảo MongoDB đang chạy cục bộ trên cổng 27017.")
    else:
        db = client["fraudDetection"]
        collection = db["fraudResults"]
        
        # --- METRICS CỦA MODEL ---
        if metrics:
            st.subheader("Hiệu suất mô hình")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(label="Độ chính xác (Accuracy)", value=f"{metrics.get('Accuracy', 0):.4f}")
            with m_col2:
                st.metric(label="Độ chuẩn xác (Fraud Precision)", value=f"{metrics.get('Fraud Precision', 0):.4f}", help="Tỷ lệ dự đoán đúng trong số các ca BỊ BÁO LÀ gian lận.")
            with m_col3:
                st.metric(label="Độ phủ (Fraud Recall)", value=f"{metrics.get('Fraud Recall', 0):.4f}", help="Tỷ lệ bắt được gian lận TRONG TỔNG SỐ các ca gian lận thực tế.")
            with m_col4:
                st.metric(label="AUPR", value=f"{metrics.get('AUPR', 0):.4f}", help="Diện tích dưới đường cong Precision-Recall. Rất quan trọng với dữ liệu mất cân bằng.")
            st.divider()

        st.subheader("Tổng quan dữ liệu")
        
        total_tx = collection.count_documents({})
        fraud_tx = collection.count_documents({"predicted_isFraud": 1.0})
        actual_fraud_tx = collection.count_documents({"actual_isFraud": 1.0})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Tổng số giao dịch", value=f"{total_tx:,}")
        with col2:
            st.metric(label="Tổng số giao dịch gian lận được dự đoán", value=f"{fraud_tx:,}")
        with col3:
            st.metric(label="Tổng số giao dịch gian lận thực tế", value=f"{actual_fraud_tx:,}")
        with col4:
            rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
            st.metric(label="Tỷ lệ cảnh báo (%)", value=f"{rate:.2f}%")

        st.divider()

        # --- VISUALIZATIONS ---
        st.subheader("Đánh giá mô hình")
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        
        def load_image(img_name):
            path = os.path.join(output_dir, img_name)
            if os.path.exists(path):
                return Image.open(path)
            return None

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            with st.container(border=True):
                st.markdown("**Ma trận nhiễu (Confusion Matrix)**")
                cm_img = load_image('confusion_matrix.png')
                if cm_img:
                    st.image(cm_img)
                    st.caption("Biểu đồ này hiển thị số lượng giao dịch bị phân loại đúng và sai. Góc phải trên là số ca bình thường bị nhận diện nhầm thành gian lận (False Positive).")
                else:
                    st.info("Không tìm thấy ảnh.")
            with st.container(border=True):
                st.markdown("**Đường cong Precision-Recall**")
                pr_img = load_image('pr_curve.png')
                if pr_img:
                    st.image(pr_img)
                    st.caption("Thể hiện sự đánh đổi giữa Precision (Độ chuẩn xác) và Recall (Độ phủ). Với dữ liệu quá mất cân bằng, đường cong này có ý nghĩa quan trọng nhất.")
                else:
                    st.info("Không tìm thấy ảnh.")

        with v_col2:
            with st.container(border=True):
                st.markdown("**Đường cong ROC**")
                roc_img = load_image('roc_curve.png')
                if roc_img:
                    st.image(roc_img)
                    st.caption("Thể hiện khả năng phân biệt giữa 2 lớp của mô hình. AUC càng gần 1, mô hình càng tốt.")
                else:
                    st.info("Không tìm thấy ảnh.")
            with st.container(border=True):
                st.markdown("**Mức độ quan trọng của các đặc trưng**")
                fi_img = load_image('feature_importances.png')
                if fi_img:
                    st.image(fi_img)
                    st.caption("Các đặc trưng được xếp hạng theo tầm quan trọng. Cột nào dài nhất thì mô hình dựa vào thông tin đó nhiều nhất để ra quyết định.")
                else:
                    st.info("Không tìm thấy ảnh.")

        st.divider()

        # --- RECENT TRANSACTIONS ---
        st.subheader("Các giao dịch rủi ro cao gần đây")
        cursor = collection.find({"predicted_isFraud": 1.0}).sort("_id", -1).limit(50)
        df_fraud = pd.DataFrame(list(cursor))
        
        if not df_fraud.empty:
            if "_id" in df_fraud.columns:
                df_fraud = df_fraud.drop(columns=["_id"])
            
            # Feature engineering on the fly to show discrepancy
            if "oldbalanceOrg" in df_fraud.columns:
                df_fraud["Sai_lech_Nguoi_gui"] = df_fraud["oldbalanceOrg"] - df_fraud["amount"] - df_fraud["newbalanceOrig"]
                df_fraud["Sai_lech_Nguoi_nhan"] = df_fraud["oldbalanceDest"] + df_fraud["amount"] - df_fraud["newbalanceDest"]
                st.markdown("Cột **Sai lệch Người gửi** và **Sai lệch Người nhận** cho thấy sự thiếu thống nhất toán học trong giao dịch (dấu hiệu mạnh của gian lận).")
            else:
                st.warning("⚠️ Đang thiếu dữ liệu số dư tài khoản trong Cơ sở dữ liệu. Dữ liệu này sẽ hiển thị sau khi Pipeline (main.py) chạy xong lần tiếp theo.")
                
            st.dataframe(df_fraud)
        else:
            st.info("Không tìm thấy giao dịch rủi ro cao nào.")

with tab_inference:
    st.subheader("Xác minh Giao dịch Thủ công")
    st.markdown("Nhập chi tiết giao dịch dưới đây để kiểm tra thông qua mô hình Random Forest đã được huấn luyện.")
    
    if not model:
        st.error("Không thể tải mô hình. Vui lòng đảm bảo pipeline đã được chạy và thư mục models/fraud_rf_model tồn tại.")
    else:
        with st.form("inference_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                tx_type = st.selectbox("Loại Giao dịch", options=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"], help="Trong tập dữ liệu Paysim, gian lận chỉ xảy ra với CASH_OUT và TRANSFER.")
                amount = st.number_input("Số tiền", min_value=0.0, value=1000.0, step=100.0, help="Tổng số tiền được thực hiện trong giao dịch.")
                is_flagged = st.selectbox("Bị cảnh báo gian lận?", options=["Không", "Có"], help="Hệ thống có gắn cờ cảnh báo giao dịch này không? (Gắn cờ cảnh báo chuyển khoản trên 200000).")
                
            with col2:
                oldbalanceOrg = st.number_input("Số dư Cũ (Người gửi)", min_value=0.0, value=1000.0, help="Số dư trong tài khoản người gửi TRƯỚC khi thực hiện giao dịch.")
                newbalanceOrig = st.number_input("Số dư Mới (Người gửi)", min_value=0.0, value=0.0, help="Số dư trong tài khoản người gửi SAU khi thực hiện giao dịch.")
                oldbalanceDest = st.number_input("Số dư Cũ (Người nhận)", min_value=0.0, value=0.0, help="Số dư trong tài khoản người nhận TRƯỚC khi thực hiện giao dịch.")
                newbalanceDest = st.number_input("Số dư Mới (Người nhận)", min_value=0.0, value=1000.0, help="Số dư trong tài khoản người nhận SAU khi thực hiện giao dịch.")
                
            submit_button = st.form_submit_button("Kiểm tra Giao dịch", type="primary")

        if submit_button:
            # Type mapping
            type_mapping = {"CASH_OUT": 0.0, "PAYMENT": 1.0, "CASH_IN": 2.0, "TRANSFER": 3.0, "DEBIT": 4.0}
            typeIndex = type_mapping.get(tx_type, 5.0)
            
            flagged = 1 if is_flagged == "Có" else 0
            
            data = [{
                "step": 1,
                "typeIndex": float(typeIndex),
                "amount": float(amount),
                "oldbalanceOrg": float(oldbalanceOrg),
                "newbalanceOrig": float(newbalanceOrig),
                "oldbalanceDest": float(oldbalanceDest),
                "newbalanceDest": float(newbalanceDest),
                "isFlaggedFraud": int(flagged)
            }]
            
            schema = StructType([
                StructField("step", IntegerType(), True),
                StructField("typeIndex", DoubleType(), True),
                StructField("amount", DoubleType(), True),
                StructField("oldbalanceOrg", DoubleType(), True),
                StructField("newbalanceOrig", DoubleType(), True),
                StructField("oldbalanceDest", DoubleType(), True),
                StructField("newbalanceDest", DoubleType(), True),
                StructField("isFlaggedFraud", IntegerType(), True)
            ])
            
            df_input = spark.createDataFrame(data, schema=schema)
            
            feature_cols = [
                "step", "typeIndex", "amount", 
                "oldbalanceOrg", "newbalanceOrig", 
                "oldbalanceDest", "newbalanceDest", 
                "isFlaggedFraud"
            ]
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
            df_assembled = assembler.transform(df_input)
            
            prediction_df = model.transform(df_assembled)
            
            result_row = prediction_df.select("prediction", "probability").collect()[0]
            prediction = result_row["prediction"]
            probability = result_row["probability"].toArray()
            fraud_prob = probability[1] * 100
            
            st.divider()
            if prediction == 1.0:
                st.error(f"🚨 **PHÁT HIỆN GIAN LẬN**! Giao dịch này được phân loại là gian lận. (Độ tin cậy: {fraud_prob:.2f}%)", icon="🚨")
            else:
                st.success(f"✅ **AN TOÀN**. Giao dịch này có vẻ hợp lệ. (Xác suất Gian lận: {fraud_prob:.2f}%)", icon="✅")
