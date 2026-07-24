import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
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


def save_confusion_matrix(y_true, y_pred, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_pr(y_true, y_score, out_dir, prefix="baseline"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(out_dir / f"{prefix}_roc.png")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(out_dir / f"{prefix}_pr.png")
    plt.close()


def find_best_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    # precision_recall_curve returns arrays where thresholds length = len(precision)-1
    if len(f1_scores) > 1:
        best_idx = np.nanargmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        best_threshold = 0.5
        best_f1 = f1_scores[0] if len(f1_scores) > 0 else 0.0
    return best_threshold, best_f1


def evaluate_at_threshold(y_true, y_score, thresh):
    y_pred = (y_score >= thresh).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report


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

    model_path = base_dir / "models" / "baseline_lr" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_path}. Run baseline.py first.")

    pipeline = joblib.load(str(model_path))

    # Probabilities on validation and test
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    out_dir = base_dir / "output" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default threshold = 0.5
    val_report_default = evaluate_at_threshold(y_val, val_proba, 0.5)
    test_report_default = evaluate_at_threshold(y_test, test_proba, 0.5)

    # Save confusion matrices
    save_confusion_matrix(y_test, (test_proba >= 0.5).astype(int), out_dir / "cm_test_default.png", "Test Confusion Matrix (0.5)")

    # ROC / PR plots on test
    plot_roc_pr(y_test, test_proba, out_dir, prefix="baseline_test")

    # Find best threshold on validation to maximize F1
    best_thresh, best_f1 = find_best_threshold(y_val, val_proba)
    tuned_test_report = evaluate_at_threshold(y_test, test_proba, best_thresh)
    save_confusion_matrix(y_test, (test_proba >= best_thresh).astype(int), out_dir / "cm_test_tuned.png", f"Test Confusion Matrix (th={best_thresh:.2f})")

    # Calibrate using validation data (prefit pipeline)
    try:
        calibrator = CalibratedClassifierCV(pipeline, cv="prefit", method="sigmoid")
        calibrator.fit(X_val, y_val)
        calib_test_proba = calibrator.predict_proba(X_test)[:, 1]
        calib_report_default = evaluate_at_threshold(y_test, calib_test_proba, 0.5)
        best_thresh_calib, best_f1_calib = find_best_threshold(y_val, calibrator.predict_proba(X_val)[:, 1])
        calib_tuned_test_report = evaluate_at_threshold(y_test, calib_test_proba, best_thresh_calib)
        plot_roc_pr(y_test, calib_test_proba, out_dir, prefix="baseline_test_calibrated")
        save_confusion_matrix(y_test, (calib_test_proba >= best_thresh_calib).astype(int), out_dir / "cm_test_calibrated_tuned.png", f"Test CM Calibrated (th={best_thresh_calib:.2f})")
    except Exception as e:
        calib_test_proba = None
        calib_report_default = None
        calib_tuned_test_report = None
        best_thresh_calib = None

    # Aggregate results into CSV
    rows = []
    rows.append({"model": "baseline", "variant": "default", "split": "test", **{f"precision_{k}": v for k, v in test_report_default.items() if k in ["0", "1"]}})
    rows.append({"model": "baseline", "variant": "tuned", "threshold": best_thresh, "split": "test", **{f"precision_{k}": v for k, v in tuned_test_report.items() if k in ["0", "1"]}})

    if calib_report_default is not None:
        rows.append({"model": "baseline", "variant": "calibrated_default", "threshold": 0.5, "split": "test", **{f"precision_{k}": v for k, v in calib_report_default.items() if k in ["0", "1"]}})
    if calib_tuned_test_report is not None:
        rows.append({"model": "baseline", "variant": "calibrated_tuned", "threshold": best_thresh_calib, "split": "test", **{f"precision_{k}": v for k, v in calib_tuned_test_report.items() if k in ["0", "1"]}})

    # Save a human-readable summary
    summary_csv = out_dir / "baseline_analysis_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print("Analysis complete.")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
