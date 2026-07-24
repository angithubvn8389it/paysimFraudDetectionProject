from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import os

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_rf_model(train_df):
    """
    Trains a Random Forest classifier using Cross Validation for hyperparameter tuning.
    """
    print("Training Random Forest model with Cross Validation...")
    # We use weightCol to handle class imbalance natively
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="classWeight", seed=42)
    
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [20, 50])
                 .addGrid(rf.maxDepth, [5, 7])
                 .build())
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
    
    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3, seed=42)
    cvModel = cv.fit(train_df)
    
    bestModel = cvModel.bestModel
    print("Training complete. Best Model Params:")
    print(f"  NumTrees: {bestModel.getNumTrees}")
    print(f"  MaxDepth: {bestModel.getOrDefault('maxDepth')}")
    
    return bestModel

def save_model(model, path):
    """
    Saves the trained Spark ML model to the specified path.
    """
    print(f"Saving model to {path}...")
    # Overwrite if exists
    model.write().overwrite().save(path)
    print("Model saved successfully.")

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from spark.preprocessing.preprocess import create_spark_session, load_data, preprocess_data
    
    spark = create_spark_session()
    df = load_data(spark)
    final_df = preprocess_data(df)
    
    # Split data
    train_df, val_df, test_df = final_df.randomSplit([0.7, 0.15, 0.15], seed=42)
    
    model = train_rf_model(train_df)
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fraud_rf_model')
    save_model(model, model_dir)
    
    spark.stop()
