import os
import sys
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)

def inspect_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Dataset not found at {RAW_DATA_PATH}")
        sys.exit(1)
        
    file_size_bytes = os.path.getsize(RAW_DATA_PATH)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"================ DATASET INSPECTION ================")
    print(f"File Path: {RAW_DATA_PATH}")
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    # Read CSV
    print("\nLoading dataset with pandas...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    total_rows, total_cols = df.shape
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {total_cols}")
    
    # Strip column names whitespace (standard in CICIDS2017)
    original_cols = list(df.columns)
    stripped_cols = [c.strip() for c in original_cols]
    has_whitespace = any(orig != strip for orig, strip in zip(original_cols, stripped_cols))
    print(f"Column names have leading/trailing spaces: {has_whitespace}")
    
    # Find Target/Label Column
    label_candidates = [c for c in original_cols if 'label' in c.lower()]
    target_col = label_candidates[0] if label_candidates else original_cols[-1]
    print(f"Identified Target/Label Column: '{target_col}' (stripped: '{target_col.strip()}')")
    
    # Class Distribution
    print("\n--- CLASS / LABEL DISTRIBUTION ---")
    class_counts = df[target_col].value_counts(dropna=False)
    class_pcts = df[target_col].value_counts(dropna=False, normalize=True) * 100
    for cls_name, count in class_counts.items():
        pct = class_pcts[cls_name]
        print(f"  * '{cls_name}': {count:,} samples ({pct:.2f}%)")
        
    # Duplicate Rows Summary
    print("\n--- DUPLICATE ROWS SUMMARY ---")
    duplicates_count = df.duplicated().sum()
    dup_pct = (duplicates_count / total_rows) * 100
    print(f"Total Duplicate Rows: {duplicates_count:,} ({dup_pct:.2f}%)")
    
    # Missing / NaN / Inf Summary
    print("\n--- MISSING & INFINITE VALUES SUMMARY ---")
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    print(f"Columns with null/NaN values: {len(cols_with_nulls)}")
    for col, n_missing in cols_with_nulls.items():
        pct = (n_missing / total_rows) * 100
        print(f"  * '{col.strip()}': {n_missing:,} missing ({pct:.4f}%)")
        
    # Check for Infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf
    print(f"\nColumns with +/- Infinite values: {len(inf_counts)}")
    for col, n_inf in inf_counts.items():
        pct = (n_inf / total_rows) * 100
        print(f"  * '{col.strip()}': {n_inf:,} infinite values ({pct:.4f}%)")
        
    # Data Types Summary
    print("\n--- DATA TYPES BREAKDOWN ---")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  * {dtype}: {count} columns")
        
    # Sample Column Names (First 15 and Last 5)
    print("\n--- SAMPLE COLUMN NAMES (Total 79 / 85) ---")
    cleaned_col_names = [c.strip() for c in original_cols]
    print(f"First 10 columns: {cleaned_col_names[:10]}")
    print(f"Last 5 columns: {cleaned_col_names[-5:]}")
    
    print("\n================ INSPECTION COMPLETE ================")

if __name__ == "__main__":
    inspect_dataset()
