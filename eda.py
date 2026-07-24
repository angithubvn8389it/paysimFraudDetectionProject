import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set PySpark Python executables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add project root to sys path
sys.path.append(os.path.dirname(__file__))

from spark.preprocessing.preprocess import create_spark_session

def run_eda():
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    spark = create_spark_session()
    
    print("Loading raw data from MongoDB...")
    # Load all data for EDA (not just the subset used in training)
    df = spark.read.format("mongodb") \
        .option("database", "fraudDetection") \
        .option("collection", "paysimData") \
        .load()
    
    # Create EDA output directory
    eda_output_dir = os.path.join(os.path.dirname(__file__), 'output', 'eda')
    os.makedirs(eda_output_dir, exist_ok=True)
    
    # 1. Basic Counts & Class Imbalance
    print("Calculating Class Distribution...")
    class_dist = df.groupBy("isFraud").count().toPandas()
    class_dist['isFraud'] = class_dist['isFraud'].map({0: 'Not Fraud', 1: 'Fraud'})
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x='isFraud', y='count', data=class_dist, palette='Set2')
    plt.title('Class Imbalance: Fraud vs Not Fraud')
    plt.yscale('log') # Log scale because non-fraud is massive
    plt.ylabel('Count (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'class_imbalance.png'), dpi=300)
    plt.close()
    
    # 2. Transaction Types Distribution
    print("Calculating Transaction Types...")
    type_dist = df.groupBy("type").count().toPandas()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='type', y='count', data=type_dist, palette='muted')
    plt.title('Distribution of Transaction Types')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'transaction_types.png'), dpi=300)
    plt.close()
    
    # 3. Fraud by Transaction Type
    print("Analyzing Fraud by Transaction Type...")
    fraud_by_type = df.filter(df.isFraud == 1).groupBy("type").count().toPandas()
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x='type', y='count', data=fraud_by_type, palette='Reds')
    plt.title('Fraudulent Transactions by Type')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'fraud_by_type.png'), dpi=300)
    plt.close()
    
    # 4. Transaction Amount Distribution (Sampled)
    print("Analyzing Transaction Amounts (Sampling 10% for speed)...")
    sample_df = df.select("amount", "isFraud").sample(withReplacement=False, fraction=0.1, seed=42).toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=sample_df, x="amount", hue="isFraud", bins=50, log_scale=True, palette="coolwarm")
    plt.title('Transaction Amount Distribution (Log Scale)')
    plt.xlabel('Amount (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'amount_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Correlation Matrix
    print("Generating Correlation Matrix...")
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud", "step"]
    # Sample again for correlation to avoid memory issues on driver
    corr_sample = df.select(num_cols).sample(withReplacement=False, fraction=0.05, seed=42).toPandas()
    corr_matrix = corr_sample.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    # 6. Boxplot for Outliers
    print("Generating Boxplots for Outlier Detection...")
    box_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    plt.figure(figsize=(12, 6))
    
    # We melt the dataframe for easier plotting in seaborn
    box_data = corr_sample[box_cols].melt(var_name='Feature', value_name='Value')
    
    # We add 1 to value so log scale doesn't fail on 0
    box_data['Value'] = box_data['Value'] + 1 
    
    sns.boxplot(x='Feature', y='Value', data=box_data, palette='Set3')
    plt.yscale('log') # Log scale because balances and amounts can be massive
    plt.title('Boxplot of Key Numerical Features (Log Scale)')
    plt.ylabel('Value (Log Scale, +1 offset)')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_dir, 'outliers_boxplot.png'), dpi=300)
    plt.close()
    
    print(f"EDA successfully completed! Check the '{eda_output_dir}' folder for the plots.")
    spark.stop()

if __name__ == "__main__":
    run_eda()
