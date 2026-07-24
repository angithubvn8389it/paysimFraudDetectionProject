import pandas as pd
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import col, spark_partition_id
import pyspark.sql.functions as F

def apply_smote_partitioned(df, feature_cols, label_col="label", random_state=42):
    """
    Applies SMOTE to a PySpark DataFrame using pandas and imbalanced-learn.
    The data is partitioned and SMOTE is applied independently within each partition.
    Note: All feature_cols must be numeric types.
    """
    
    # We need to ensure the schema matches what pandas will return.
    schema = df.select(feature_cols + [label_col]).schema
    
    def smote_pandas(iterator):
        for pdf in iterator:
            if pdf.empty:
                continue
                
            X = pdf[feature_cols]
            y = pdf[label_col]
            
            # Check if we have both classes in this partition
            if len(y.unique()) > 1:
                # Get the count of minority class
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                
                # SMOTE requires n_neighbors <= n_samples. Default n_neighbors is 5.
                # So we need at least 6 samples in the minority class in this partition.
                if min_class_count > 6:
                    sm = SMOTE(random_state=random_state)
                    X_res, y_res = sm.fit_resample(X, y)
                    
                    res_pdf = pd.DataFrame(X_res, columns=feature_cols)
                    res_pdf[label_col] = y_res
                    yield res_pdf
                else:
                    # Not enough samples for SMOTE in this partition, yield original
                    yield pdf
            else: 
                # Only one class present, yield original
                yield pdf

    # Repartition to ensure decent size partitions for SMOTE
    # Since fraud is rare, fewer partitions might group more fraud cases together
    # but we also don't want out-of-memory.
    df_repart = df.repartition(10)
    
    resampled_df = df_repart.select(feature_cols + [label_col]).mapInPandas(smote_pandas, schema=schema)
    
    return resampled_df
