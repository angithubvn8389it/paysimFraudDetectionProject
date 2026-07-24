import time
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import baseline


def measure_baseline(base_dir, out_dir):
    timings = {}

    # Data load + feature derivation
    start = time.perf_counter()
    train, val, test = baseline.load_data(base_dir)
    timings['data_load'] = time.perf_counter() - start

    # Feature matrix creation
    start = time.perf_counter()
    feature_cols = [
        "step",
        "typeIndex",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "errorBalanceOrig",
        "errorBalanceDest",
        "amountToBalanceRatio",
    ]
    X_train, y_train = baseline.make_features_label(train, feature_cols)
    X_val, y_val = baseline.make_features_label(val, feature_cols)
    X_test, y_test = baseline.make_features_label(test, feature_cols)
    timings['feature_matrix_creation'] = time.perf_counter() - start

    # Model init
    start = time.perf_counter()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    timings['model_init'] = time.perf_counter() - start

    # Training
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    timings['training'] = time.perf_counter() - start

    # Evaluation
    start = time.perf_counter()
    val_metrics = baseline.evaluate_model(pipeline, X_val, y_val)
    test_metrics = baseline.evaluate_model(pipeline, X_test, y_test)
    timings['evaluation'] = time.perf_counter() - start

    # Saving artifacts
    start = time.perf_counter()
    model_dir = base_dir / "models" / "baseline_lr"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(model_dir / "model.joblib"))
    metrics_path = out_dir / "metrics_baseline.txt"
    with open(metrics_path, "w") as f:
        f.write("Validation metrics:\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTest metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
    timings['saving'] = time.perf_counter() - start

    return timings


def measure_random_forest(base_dir, out_dir):
    timings = {}
    try:
        # Import spark training helpers
        from spark.preprocessing.preprocess import create_spark_session, load_data, preprocess_data
        from spark.training.trainingModel import train_rf_model, save_model
        import os

        # Spark session creation
        start = time.perf_counter()
        spark = create_spark_session()
        timings['spark_session_create'] = time.perf_counter() - start

        # Load data
        start = time.perf_counter()
        df = load_data(spark)
        timings['spark_load_data'] = time.perf_counter() - start

        # Preprocess
        start = time.perf_counter()
        final_df = preprocess_data(df)
        timings['spark_preprocess'] = time.perf_counter() - start

        # Split
        start = time.perf_counter()
        train_df, val_df, test_df = final_df.randomSplit([0.7, 0.15, 0.15], seed=42)
        timings['spark_split'] = time.perf_counter() - start

        # Calculate class weights and assemble features (match main.py)
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql.functions import when, col
        feature_cols = [
            "step", "typeIndex", "amount",
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
            "isFlaggedFraud", "errorBalanceOrig", "errorBalanceDest", "amountToBalanceRatio"
        ]

        start = time.perf_counter()
        fraud_count = train_df.filter(train_df.label == 1).count()
        non_fraud_count = train_df.filter(train_df.label == 0).count()
        total_count = train_df.count()
        weight_fraud = total_count / (2.0 * fraud_count) if fraud_count > 0 else 1.0
        weight_non_fraud = total_count / (2.0 * non_fraud_count) if non_fraud_count > 0 else 1.0
        train_df = train_df.withColumn("classWeight", when(col("label") == 1, weight_fraud).otherwise(weight_non_fraud))

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        train_df = assembler.transform(train_df)
        val_df = assembler.transform(val_df)
        test_df = assembler.transform(test_df)
        # cache training data
        train_df.cache()
        timings['spark_feature_assembly'] = time.perf_counter() - start

        # Train (this includes cross-validation inside)
        start = time.perf_counter()
        model = train_rf_model(train_df)
        timings['spark_training'] = time.perf_counter() - start

        # Save model
        start = time.perf_counter()
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'fraud_rf_model')
        save_model(model, model_dir)
        timings['spark_save'] = time.perf_counter() - start

        # Stop spark
        try:
            spark.stop()
        except Exception:
            pass

    except Exception as e:
        # If Spark or imports fail, record the error
        timings['spark_error'] = str(e)

    return timings


def main():
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "output"
    out_dir.mkdir(exist_ok=True)

    all_timings = {}

    # Baseline timings
    try:
        baseline_timings = measure_baseline(base_dir, out_dir)
        all_timings['baseline'] = baseline_timings
    except Exception as e:
        all_timings['baseline_error'] = str(e)

    # Random Forest timings (Spark)
    rf_timings = measure_random_forest(base_dir, out_dir)
    all_timings['random_forest'] = rf_timings

    # Write timings to a single txt file
    timing_file = out_dir / "timing_models.txt"
    with open(timing_file, "w") as f:
        for model_name, timings in all_timings.items():
            f.write(f"Model: {model_name}\n")
            if isinstance(timings, dict):
                for k, v in timings.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {timings}\n")
            f.write("\n")

    print(f"Timing saved to: {timing_file}")


if __name__ == '__main__':
    main()
