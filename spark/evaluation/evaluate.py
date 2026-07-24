from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_model(model, test_df):
    """
    Evaluates the trained model on test data, calculating AUROC, AUPR, and various
    other classification metrics (precision, recall, F1) overall and for the fraud class.
    """
    print("Evaluating model...")
    predictions = model.transform(test_df)
    
    # AUROC and AUPR
    evaluator_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auroc = evaluator_roc.evaluate(predictions)
    
    evaluator_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    aupr = evaluator_pr.evaluate(predictions)
    
    # Precision, Recall, F1
    evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    
    precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
    recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
    f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
    accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
    
    # Specific to Fraud class (label 1.0)
    precision_fraud = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "precisionByLabel", evaluator_multi.metricLabel: 1.0})
    recall_fraud = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "recallByLabel", evaluator_multi.metricLabel: 1.0})
    f1_fraud = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "fMeasureByLabel", evaluator_multi.metricLabel: 1.0})
    
    # Calculate Fraud F2-Score manually (weights recall higher than precision)
    if (4 * precision_fraud + recall_fraud) != 0:
        f2_fraud = (5 * precision_fraud * recall_fraud) / (4 * precision_fraud + recall_fraud)
    else:
        f2_fraud = 0.0
    
    metrics = {
        "AUROC": auroc,
        "AUPR": aupr,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1-Score": f1,
        "Fraud Precision": precision_fraud,
        "Fraud Recall": recall_fraud,
        "Fraud F1-Score": f1_fraud,
        "Fraud F2-Score": f2_fraud
    }
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("--------------------------\n")
    
    return predictions, metrics

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from spark.preprocessing.preprocess import create_spark_session, load_data, preprocess_data
    from pyspark.ml.classification import RandomForestClassificationModel
    
    spark = create_spark_session()
    df = load_data(spark)
    final_df = preprocess_data(df)
    
    # Split data to get the test set (must use same seed as training)
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
    
    # Load model
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'fraud_rf_model')
    model = RandomForestClassificationModel.load(model_dir)
    
    predictions, metrics = evaluate_model(model, test_df)
    
    spark.stop()
