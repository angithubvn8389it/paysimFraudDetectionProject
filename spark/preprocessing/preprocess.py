from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def create_spark_session():
    """
    Creates and configures a SparkSession with MongoDB connector settings.
    """
    # Adding mongo-spark connector package
    spark = SparkSession.builder \
        .appName("PaysimFraudDetection") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.read.connection.uri", "mongodb+srv://andang32822:iXFW6YpPRR8Mpt89@mongodbhostingcluster.wdfv27d.mongodb.net/") \
        .config("spark.mongodb.write.connection.uri", "mongodb+srv://andang32822:iXFW6YpPRR8Mpt89@mongodbhostingcluster.wdfv27d.mongodb.net/") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "120s") \
        .getOrCreate()
    return spark

def load_data(spark):
    """
    Loads fraudulent and normal transaction data from MongoDB.
    """
    print("Loading data from MongoDB...")
    # The MongoDB connector natively pushes down filters and limits to the database!
    base_df = spark.read.format("mongodb") \
        .option("database", "fraudDetection") \
        .option("collection", "paysimData") \
        .load()
        
    # Fetch all fraud cases
    fraud_df = base_df.filter(base_df.isFraud == 1)
    
    # Fetch a subset of normal cases
    normal_df = base_df.filter(base_df.isFraud == 0).limit(292000)
    
    # Combine them
    df = fraud_df.unionByName(normal_df)
    return df

def preprocess_data(df):
    """
    Preprocesses the data: handles missing columns, casts data types,
    and encodes categorical features for the ML pipeline.
    """
    print("Preprocessing data...")
    # Drop unnecessary _id column added by mongo
    if "_id" in df.columns:
        df = df.drop("_id")
        
    # Cast numerical columns just in case
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud", "step"]
    for col_name in num_cols:
        if col_name in df.columns:
            if col_name in ["isFraud", "isFlaggedFraud", "step"]:
                df = df.withColumn(col_name, df[col_name].cast(IntegerType()))
            else:
                df = df.withColumn(col_name, df[col_name].cast(DoubleType()))

    # Encode categorical feature 'type' using manual mapping to avoid StringIndexer.fit() which triggers eager evaluation and crashes
    from pyspark.sql.functions import col, when
    preprocessed_df = df.withColumn("typeIndex", 
        when(col("type") == "CASH_OUT", 0.0)
        .when(col("type") == "PAYMENT", 1.0)
        .when(col("type") == "CASH_IN", 2.0)
        .when(col("type") == "TRANSFER", 3.0)
        .when(col("type") == "DEBIT", 4.0)
        .otherwise(5.0)
    )

    # --- NEW FEATURE ENGINEERING ---
    # 1. Error in Origin Balance (often mathematically incorrect for fraud transactions)
    preprocessed_df = preprocessed_df.withColumn("errorBalanceOrig", col("newbalanceOrig") + col("amount") - col("oldbalanceOrg"))
    # 2. Error in Destination Balance
    preprocessed_df = preprocessed_df.withColumn("errorBalanceDest", col("oldbalanceDest") + col("amount") - col("newbalanceDest"))
    # 3. Amount to Balance Ratio (adds epsilon 1e-6 to avoid division by zero)
    preprocessed_df = preprocessed_df.withColumn("amountToBalanceRatio", col("amount") / (col("oldbalanceOrg") + 1e-6))


    # nameOrig and nameDest have high cardinality, we usually drop them or hash them.
    # For this project, we'll drop them to keep the model simple and avoid overfitting.
    cols_to_drop = ["nameOrig", "nameDest", "type"]
    preprocessed_df = preprocessed_df.drop(*cols_to_drop)

    # Rename isFraud to label
    final_df = preprocessed_df.withColumnRenamed("isFraud", "label")
    
    print("Preprocessing complete.")
    return final_df

if __name__ == "__main__":
    spark = create_spark_session()
    df = load_data(spark)
    final_df = preprocess_data(df)
    final_df.show(5)
    spark.stop()
