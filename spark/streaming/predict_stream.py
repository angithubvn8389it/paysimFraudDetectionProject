import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col, from_json, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel

# Set PySpark Python executables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

# MongoDB Configuration
MONGO_URI = "mongodb+srv://andang32822:iXFW6YpPRR8Mpt89@mongodbhostingcluster.wdfv27d.mongodb.net/"

def create_spark_session():
    # We need both Kafka and MongoDB packages
    spark = SparkSession.builder \
        .appName("PaysimFraudDetectionStreaming") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .getOrCreate()
    return spark

def process_batch(df, epoch_id):
    """
    Process each micro-batch: write the predictions to MongoDB.
    """
    if df.count() == 0:
        return
        
    print(f"Processing batch {epoch_id} with {df.count()} records.")
    
    # MongoDB cannot natively store Spark ML Vectors. Convert probability to Array of floats.
    from pyspark.ml.functions import vector_to_array
    
    # Select important columns to save
    cols_to_save = [
        "step", "type", "amount", "nameOrig", 
        "oldbalanceOrg", "newbalanceOrig", "nameDest",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud", "prediction", "probability"
    ]
    
    final_output_df = df.select([c for c in cols_to_save if c in df.columns])
    
    if "probability" in final_output_df.columns:
        final_output_df = final_output_df.withColumn("probability", vector_to_array(col("probability")))
        
    final_output_df.write.format("mongodb") \
        .option("database", "fraudDetection") \
        .option("collection", "fraudResultsStream") \
        .mode("append") \
        .save()

def main():
    print("--- Starting Real-Time Fraud Prediction Stream ---")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fraud_rf_model')
    print(f"Loading Random Forest Model from {model_dir}...")
    try:
        model = RandomForestClassificationModel.load(model_dir)
    except Exception as e:
        print(f"Failed to load model. Did you run main.py to train it? Error: {e}")
        return

    # Define the schema of the incoming JSON data from Kafka
    schema = StructType([
        StructField("step", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("nameOrig", StringType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
        StructField("nameDest", StringType(), True),
        StructField("oldbalanceDest", DoubleType(), True),
        StructField("newbalanceDest", DoubleType(), True),
        StructField("isFraud", IntegerType(), True),
        StructField("isFlaggedFraud", IntegerType(), True)
    ])

    print("Connecting to Kafka stream...")
    # Read stream from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "paysim-transactions") \
        .option("startingOffsets", "latest") \
        .load()
        
    # Convert Kafka value (binary) to String, then parse JSON using schema
    json_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Preprocessing
    # 1. Map type to typeIndex
    processed_stream = json_stream.withColumn("typeIndex", 
        when(col("type") == "CASH_OUT", 0.0)
        .when(col("type") == "PAYMENT", 1.0)
        .when(col("type") == "CASH_IN", 2.0)
        .when(col("type") == "TRANSFER", 3.0)
        .when(col("type") == "DEBIT", 4.0)
        .otherwise(5.0)
    )
    
    # 2. Add New Features (Math/Balance Errors & Ratios)
    processed_stream = processed_stream.withColumn("errorBalanceOrig", col("newbalanceOrig") + col("amount") - col("oldbalanceOrg"))
    processed_stream = processed_stream.withColumn("errorBalanceDest", col("oldbalanceDest") + col("amount") - col("newbalanceDest"))
    processed_stream = processed_stream.withColumn("amountToBalanceRatio", col("amount") / (col("oldbalanceOrg") + 1e-6))
    
    # 3. Assemble features
    feature_cols = [
        "step", "typeIndex", "amount", 
        "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", 
        "isFlaggedFraud",
        "errorBalanceOrig", "errorBalanceDest", "amountToBalanceRatio"
    ]
    
    # Fill nulls with 0 to prevent vector assembler from dropping rows (streaming safety)
    processed_stream = processed_stream.fillna(0.0, subset=feature_cols)
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    assembled_stream = assembler.transform(processed_stream)
    
    # 3. Predict using the loaded model
    predictions_stream = model.transform(assembled_stream)
    
    print("Starting Streaming Query to MongoDB...")
    # Write predictions to MongoDB using foreachBatch
    query = predictions_stream.writeStream \
        .outputMode("append") \
        .foreachBatch(process_batch) \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    main()
