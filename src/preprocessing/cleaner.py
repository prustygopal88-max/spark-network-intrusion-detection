import os
import re
import numpy as np
import pandas as pd

LABEL_MAPPING = {
    "BENIGN": 0,
    "PortScan": 1
}

def clean_column_name(col_name: str) -> str:
    """
    Standardizes column names for PySpark SQL and MLlib compatibility:
    - Strips leading/trailing whitespace
    - Replaces '/', '\', spaces, and '-' with single underscores
    - Removes non-alphanumeric characters
    - Converts to lowercase
    """
    cleaned = col_name.strip()
    cleaned = re.sub(r"[\s/\\-]+", "_", cleaned)
    cleaned = re.sub(r"[^\w]", "", cleaned)
    return cleaned.lower()

def clean_cicids_dataset(raw_csv_path: str):
    """
    Performs end-to-end data cleaning on raw CICIDS2017 CSV:
    1. Standardizes column names
    2. Maps categorical target label ('BENIGN' -> 0, 'PortScan' -> 1)
    3. Handles and counts duplicate rows
    4. Replaces infinite values (+/- inf) with NaN and drops null rows
    5. Detects and removes zero-variance (constant) columns
    6. Ensures proper numeric float64 casting on all feature columns
    
    Returns:
        cleaned_df (pd.DataFrame): Processed data ready for Spark
        metadata (dict): Comprehensive tracking metrics
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found: {raw_csv_path}")

    # Load raw dataset
    raw_df = pd.read_csv(raw_csv_path, low_memory=False)
    raw_row_count, raw_col_count = raw_df.shape

    # Identify label column before renaming
    label_candidates = [c for c in raw_df.columns if "label" in c.lower()]
    raw_label_col = label_candidates[0] if label_candidates else raw_df.columns[-1]

    # Calculate class distribution before preprocessing
    raw_class_counts = raw_df[raw_label_col].value_counts().to_dict()
    raw_class_pcts = (raw_df[raw_label_col].value_counts(normalize=True) * 100).round(2).to_dict()

    # Step 1: Clean column names
    col_mapping = {col: clean_column_name(col) for col in raw_df.columns}
    df = raw_df.rename(columns=col_mapping)
    target_col = col_mapping[raw_label_col]

    # Step 2: Encode Target / Label
    # Filter only expected labels (handling any potential corrupted rows)
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[df[target_col].isin(LABEL_MAPPING.keys())].copy()
    df[target_col] = df[target_col].map(LABEL_MAPPING).astype(int)

    # Step 3: Remove duplicate rows
    duplicates_detected = int(df.duplicated().sum())
    df = df.drop_duplicates()
    post_dedup_count = len(df)

    # Step 4: Handle infinite and missing values
    feature_columns = [c for c in df.columns if c != target_col]
    
    # Count infinite entries across numeric features
    inf_mask = np.isinf(df[feature_columns].values)
    total_inf_values = int(np.sum(inf_mask))
    
    # Replace inf with NaN and drop missing rows
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_rows_before = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    cleaned_row_count = len(df)
    missing_and_inf_rows_dropped = post_dedup_count - cleaned_row_count

    # Step 5: Identify and drop constant (zero-variance) columns
    constant_columns = [col for col in feature_columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_columns)
    
    # Final remaining features
    final_feature_cols = [c for c in df.columns if c != target_col]
    
    # Ensure all feature columns are float64 for Spark MLlib VectorAssembler
    for col in final_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64)

    # Calculate class distribution after preprocessing
    clean_counts_raw = df[target_col].value_counts().to_dict()
    clean_class_counts = {
        "BENIGN (0)": int(clean_counts_raw.get(0, 0)),
        "PortScan (1)": int(clean_counts_raw.get(1, 0))
    }
    clean_class_pcts = {
        "BENIGN (0)": round((clean_counts_raw.get(0, 0) / cleaned_row_count) * 100, 2),
        "PortScan (1)": round((clean_counts_raw.get(1, 0) / cleaned_row_count) * 100, 2)
    }

    metadata = {
        "raw_rows": raw_row_count,
        "raw_cols": raw_col_count,
        "cleaned_rows": cleaned_row_count,
        "cleaned_cols": df.shape[1],
        "feature_count": len(final_feature_cols),
        "target_col": target_col,
        "label_mapping": LABEL_MAPPING,
        "duplicates_removed": duplicates_detected,
        "infinite_values_found": total_inf_values,
        "missing_and_inf_rows_removed": missing_and_inf_rows_dropped,
        "constant_columns_dropped": constant_columns,
        "class_distribution_before": {
            "counts": raw_class_counts,
            "percentages": raw_class_pcts
        },
        "class_distribution_after": {
            "counts": clean_class_counts,
            "percentages": clean_class_pcts
        },
        "feature_columns": final_feature_cols
    }

    return df, metadata

if __name__ == "__main__":
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "raw", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    )
    cleaned_data, meta = clean_cicids_dataset(raw_path)
    print("Cleaning complete. Sample summary:")
    print(f"Original: {meta['raw_rows']} -> Cleaned: {meta['cleaned_rows']}")
    print(f"Features: {meta['feature_count']} (dropped {len(meta['constant_columns_dropped'])} constant cols)")
    print(f"Duplicates removed: {meta['duplicates_removed']}")
    print(f"Class distribution after: {meta['class_distribution_after']['counts']}")
