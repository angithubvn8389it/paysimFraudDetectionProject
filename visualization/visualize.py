import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

def generate_visualizations(predictions, model):
    print("Generating visualizations...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for confusion matrix
    # Note: collecting predictions to driver node (this is fine for small test sets, but be careful with large datasets)
    preds_df = predictions.select("label", "prediction", "probability").toPandas()
    # Extract probability for the positive class (index 1)
    preds_df['prob_fraud'] = preds_df['probability'].apply(lambda x: x[1])
    
    # 1. Confusion Matrix
    cm = confusion_matrix(preds_df['label'], preds_df['prediction'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Random Forest')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # 2. Feature Importances
    # Extract feature names from VectorAssembler in the future, for now we hardcode based on preprocess
    feature_names = [
        "step", "typeIndex", "amount", 
        "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", 
        "isFlaggedFraud",
        "errorBalanceOrig", "errorBalanceDest", "amountToBalanceRatio"
    ]
    
    # For RandomForest, the actual model is inside the pipeline or returned directly if fit directly
    importances = model.featureImportances.toArray()
    
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title('Feature Importances - Random Forest')
    plt.tight_layout()
    fi_path = os.path.join(output_dir, 'feature_importances.png')
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved Feature Importances to {fi_path}")
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(preds_df['label'], preds_df['prob_fraud'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(preds_df['label'], preds_df['prob_fraud'])
    pr_auc = average_precision_score(preds_df['label'], preds_df['prob_fraud'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved Precision-Recall Curve to {pr_path}")

if __name__ == "__main__":
    pass # Usually called from main.py
