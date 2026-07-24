"""
Main entry point for the Paysim Fraud Detection Pipeline.
This script orchestrates the entire process:
1. Spark Session Initialization
2. Data Loading & Preprocessing
3. Model Training (Random Forest)
4. Model Evaluation
5. Visualization Generation
6. Saving Predictions to MongoDB
"""
import os
import sys

# Set PySpark Python executables to the current Python executable
# This fixes the "Python worker failed to connect back" error
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

# Add project root to sys path
sys.path.append(os.path.dirname(__file__))

from spark.preprocessing.preprocess import create_spark_session, load_data, preprocess_data
from pyspark.ml.feature import VectorAssembler
from spark.training.trainingModel import train_rf_model, save_model
from spark.evaluation.evaluate import evaluate_model
from visualization.visualize import generate_visualizations

def main():
    """
    Executes the end-to-end fraud detection pipeline.
    Initializes Spark, loads and preprocesses data, trains a Random Forest model,
    evaluates its performance, and saves the results.
    """
    print("--- Starting Paysim Fraud Detection Pipeline ---")
    
    # 1. Initialize Spark Session
    print("\n[1/5] Initializing Spark...")
    spark = create_spark_session()
    
    # 2. Load and Preprocess Data
    print("\n[2/5] Loading and Preprocessing Data...")
    df = load_data(spark)
    final_df = preprocess_data(df)

    # Save the processed data to local disk as Parquet for future use/inspection
    # processed_data_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'processed_data.parquet')
    # print(f"Saving processed data to {processed_data_path}...")
    # final_df.write.mode("overwrite").parquet(processed_data_path)

    # Split Data
    print("Splitting data into 70% train, 15% validation, and 15% test...")
    train_df, val_df, test_df = final_df.randomSplit([0.7, 0.15, 0.15], seed=42)
    
    print("Saving train, val, and test splits to CSV...")
    split_output_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'splits')
    os.makedirs(split_output_dir, exist_ok=True)
    
    train_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(os.path.join(split_output_dir, "train_data"))
    val_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(os.path.join(split_output_dir, "val_data"))
    test_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(os.path.join(split_output_dir, "test_data"))
    
    print("Calculating Class Weights for imbalanced data...")
    feature_cols = [
        "step", "typeIndex", "amount", 
        "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", 
        "isFlaggedFraud",
        "errorBalanceOrig", "errorBalanceDest", "amountToBalanceRatio"
    ]
    
    # Calculate weights to balance the classes
    fraud_count = train_df.filter(train_df.label == 1).count()
    non_fraud_count = train_df.filter(train_df.label == 0).count()
    total_count = train_df.count()
    
    weight_fraud = total_count / (2.0 * fraud_count)
    weight_non_fraud = total_count / (2.0 * non_fraud_count)
    
    from pyspark.sql.functions import when, col
    train_df = train_df.withColumn("classWeight", when(col("label") == 1, weight_fraud).otherwise(weight_non_fraud))
    
    # Assemble Features
    print("Assembling features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    train_df = assembler.transform(train_df)
    val_df = assembler.transform(val_df)
    test_df = assembler.transform(test_df)
    
    # Cache the data before training! CrossValidator trains 12 models. 
    # Without caching, Spark will re-run the entire pipeline (including reading from MongoDB and running SMOTE) 12 times!
    print("Caching training data...")
    train_df.cache()
    
    # 3. Train Model
    print("\n[3/5] Training Model...")
    model = train_rf_model(train_df)
    
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'fraud_rf_model')
    save_model(model, model_dir)
    
    # 4. Evaluate Model
    print("\n[4/5] Evaluating Model on Training Set...")
    train_predictions, train_metrics = evaluate_model(model, train_df)
    print(f"Training Metrics: {train_metrics}")

    print("\nEvaluating Model on Validation Set...")
    val_predictions, val_metrics = evaluate_model(model, val_df)
    print(f"Validation Metrics: {val_metrics}")
    
    print("\nEvaluating Model on Test Set...")
    predictions, metrics = evaluate_model(model, test_df)
    
    # Save metrics to CSV for comparison
    import pandas as pd
    import json
    metrics_df = pd.DataFrame([
        {"dataset": "train", **train_metrics},
        {"dataset": "validation", **val_metrics},
        {"dataset": "test", **metrics}
    ])
    csv_metrics_path = os.path.join(os.path.dirname(__file__), 'output', 'metrics.csv')
    os.makedirs(os.path.dirname(csv_metrics_path), exist_ok=True)
    metrics_df.to_csv(csv_metrics_path, index=False)
    print(f"Metrics saved to {csv_metrics_path}")
    
    # Save test metrics to JSON for the dashboard
    metrics_path = os.path.join(os.path.dirname(__file__), 'output', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 5. Visualize Results
    print("\n[5/5] Generating Visualizations...")
    generate_visualizations(predictions, model)
    
    # 6. Save Predictions to MongoDB
    print("\n[6/6] Saving Predictions to MongoDB...")
    output_df = predictions.withColumnRenamed("prediction", "predicted_isFraud") \
                           .withColumnRenamed("label", "actual_isFraud")
    
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql.functions import col
    
    # We select only a subset of important columns to save to avoid bloating the database
    cols_to_save = [
        "step", "typeIndex", "amount", 
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "actual_isFraud", "predicted_isFraud", "probability"
    ]
    final_output_df = output_df.select([col for col in cols_to_save if col in output_df.columns])
    
    # MongoDB cannot natively store Spark ML Vectors. We must convert the probability Vector to an Array of floats.
    if "probability" in final_output_df.columns:
        final_output_df = final_output_df.withColumn("probability", vector_to_array(col("probability")))
    
    final_output_df.write.format("mongodb") \
        .option("database", "fraudDetection") \
        .option("collection", "fraudResults") \
        .mode("overwrite") \
        .save()
        
    print("Predictions successfully saved to MongoDB (fraudDetection.fraudResults)!")
    
    print("\n--- Pipeline Completed Successfully ---")
    spark.stop()

if __name__ == "__main__":
    main()
