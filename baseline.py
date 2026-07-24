import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib


def read_split_dir(dir_path):
    files = glob.glob(os.path.join(dir_path, "part-*.csv"))
    if not files:
        raise FileNotFoundError(f"No part-*.csv files in {dir_path}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def add_derived_features(df):
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["amountToBalanceRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-6)
    return df


def load_data(base_dir):
    base = Path(base_dir)
    train_dir = base / "data" / "processed" / "splits" / "train_data"
    val_dir = base / "data" / "processed" / "splits" / "val_data"
    test_dir = base / "data" / "processed" / "splits" / "test_data"

    train = read_split_dir(str(train_dir))
    val = read_split_dir(str(val_dir))
    test = read_split_dir(str(test_dir))

    train = add_derived_features(train)
    val = add_derived_features(val)
    test = add_derived_features(test)

    return train, val, test


def make_features_label(df, feature_cols):
    X = df[feature_cols].fillna(0).values
    y = df["label"].astype(int).values
    return X, y


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    res = {
        "accuracy": metrics.accuracy_score(y, y_pred),
        "precision": metrics.precision_score(y, y_pred, zero_division=0),
        "recall": metrics.recall_score(y, y_pred, zero_division=0),
        "f1": metrics.f1_score(y, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y)) == 2:
        try:
            res["roc_auc"] = metrics.roc_auc_score(y, y_proba)
        except Exception:
            res["roc_auc"] = np.nan
    else:
        res["roc_auc"] = np.nan
    return res


def main():
    base_dir = Path(__file__).resolve().parent
    train, val, test = load_data(base_dir)

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

    X_train, y_train = make_features_label(train, feature_cols)
    X_val, y_val = make_features_label(val, feature_cols)
    X_test, y_test = make_features_label(test, feature_cols)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    val_metrics = evaluate_model(pipeline, X_val, y_val)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    out_dir = base_dir / "output"
    out_dir.mkdir(exist_ok=True)

    metrics_df = pd.DataFrame([
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])

    metrics_csv = out_dir / "metrics_baseline.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    model_dir = base_dir / "models" / "baseline_lr"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(model_dir / "model.joblib"))

    print("Baseline training complete.")
    print(f"Metrics saved to: {metrics_csv}")
    print(f"Model saved to: {model_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
