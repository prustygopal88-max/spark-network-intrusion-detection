import os
import sys
import json
import time
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.spark_session import get_spark_session, stop_spark_session
from src.preprocessing.cleaner import clean_cicids_dataset
from src.preprocessing.feature_pipeline import fit_feature_pipeline

def run_pipeline():
    start_time = time.time()
    print("================ STAGE 3: DATA PREPROCESSING & FEATURE PIPELINE ================\n")

    raw_csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    models_dir = os.path.join(PROJECT_ROOT, "results", "models")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Clean raw dataset
    print("[1/6] Cleaning raw CICIDS2017 dataset...")
    cleaned_df, metadata = clean_cicids_dataset(raw_csv_path)
    print(f"      Raw rows: {metadata['raw_rows']:,} -> Cleaned rows: {metadata['cleaned_rows']:,}")
    print(f"      Duplicates removed: {metadata['duplicates_removed']:,}")
    print(f"      Missing/Inf rows dropped: {metadata['missing_and_inf_rows_removed']:,}")
    print(f"      Zero-variance features removed: {len(metadata['constant_columns_dropped'])}")
    print(f"      Retained predictive features: {metadata['feature_count']}")

    # 2. Save cleaned dataset and sample for fast dashboard loading
    print("\n[2/6] Exporting cleaned dataset and lightweight preview sample...")
    cleaned_sample_path = os.path.join(processed_dir, "cleaned_sample.csv")
    cleaned_df.head(10000).to_csv(cleaned_sample_path, index=False)
    print(f"      Cleaned preview sample (10,000 rows) saved to: {cleaned_sample_path}")

    # 3. Initialize PySpark session
    print("\n[3/6] Initializing local SparkSession...")
    spark = get_spark_session(app_name="CICIDS2017-Preprocessing")
    print(f"      Spark version: {spark.version}")

    try:
        # Convert cleaned pandas dataframe to Spark DataFrame
        print("      Converting cleaned dataset to Spark DataFrame...")
        spark_df = spark.createDataFrame(cleaned_df)
        spark_df.cache()
        total_spark_rows = spark_df.count()
        print(f"      Spark DataFrame row count verified: {total_spark_rows:,}")

        # 4. Split into Training (80%) and Testing (20%) sets
        print("\n[4/6] Splitting data into Train (80%) and Test (20%) with seed=42...")
        train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)
        train_df.cache()
        test_df.cache()

        train_count = train_df.count()
        test_count = test_df.count()
        print(f"      Train set rows: {train_count:,} ({train_count / total_spark_rows * 100:.2f}%)")
        print(f"      Test set rows:  {test_count:,} ({test_count / total_spark_rows * 100:.2f}%)")

        # Check train and test class distributions
        train_class_counts = train_df.groupBy("label").count().toPandas().set_index("label")["count"].to_dict()
        test_class_counts = test_df.groupBy("label").count().toPandas().set_index("label")["count"].to_dict()

        # 5. Fit Spark MLlib Feature Pipeline on train set
        print("\n[5/6] Fitting Spark MLlib Feature Pipeline (VectorAssembler + StandardScaler)...")
        feature_cols = metadata["feature_columns"]
        pipeline_model_path = os.path.join(models_dir, "feature_pipeline_model")
        pipeline_model = fit_feature_pipeline(train_df, feature_cols, save_path=pipeline_model_path)

        # Transform train and test DataFrames
        print("      Transforming train and test sets with fitted pipeline...")
        train_transformed = pipeline_model.transform(train_df)
        test_transformed = pipeline_model.transform(test_df)

        # Verify feature vector dimensionality
        first_row = train_transformed.select("features", "label").first()
        vector_size = first_row["features"].size
        print(f"      Final feature vector dimension: {vector_size}")

        # 6. Save processed datasets
        print("\n[6/6] Saving processed train and test sets in data/processed/...")
        train_output_path = os.path.join(processed_dir, "train.parquet")
        test_output_path = os.path.join(processed_dir, "test.parquet")
        
        train_transformed.write.mode("overwrite").parquet(train_output_path)
        print(f"      Train dataset saved to: {train_output_path}")

        test_transformed.write.mode("overwrite").parquet(test_output_path)
        print(f"      Test dataset saved to:  {test_output_path}")

        # Update metadata report
        metadata["train_rows"] = train_count
        metadata["test_rows"] = test_count
        metadata["train_class_counts"] = {
            "BENIGN (0)": int(train_class_counts.get(0, 0)),
            "PortScan (1)": int(train_class_counts.get(1, 0))
        }
        metadata["test_class_counts"] = {
            "BENIGN (0)": int(test_class_counts.get(0, 0)),
            "PortScan (1)": int(test_class_counts.get(1, 0))
        }
        metadata["final_vector_size"] = vector_size
        metadata["train_path"] = train_output_path
        metadata["test_path"] = test_output_path
        metadata["pipeline_model_path"] = pipeline_model_path
        metadata["pipeline_execution_time_seconds"] = round(time.time() - start_time, 2)

        # Save JSON report
        json_report_path = os.path.join(results_dir, "preprocessing_report.json")
        with open(json_report_path, "w") as jf:
            json.dump(metadata, jf, indent=4)
        print(f"\nSaved structured metadata to: {json_report_path}")

        # Save Markdown report
        md_report_path = os.path.join(results_dir, "preprocessing_report.md")
        with open(md_report_path, "w") as mf:
            mf.write(f"""# CICIDS2017 Preprocessing & Spark Feature Pipeline Report

## Overview
- **Raw Rows:** {metadata['raw_rows']:,}
- **Cleaned Rows:** {metadata['cleaned_rows']:,}
- **Duplicates Removed:** {metadata['duplicates_removed']:,}
- **Missing / Infinite Rows Dropped:** {metadata['missing_and_inf_rows_removed']:,}
- **Constant Columns Removed:** {len(metadata['constant_columns_dropped'])} ({', '.join(metadata['constant_columns_dropped'])})
- **Features Retained:** {metadata['feature_count']}
- **Final Feature Vector Dimension:** {metadata['final_vector_size']}
- **Execution Time:** {metadata['pipeline_execution_time_seconds']} seconds

## Class Distribution Before vs After

| Class | Original Count | Original % | Cleaned Count | Cleaned % |
|:---|:---:|:---:|:---:|:---:|
| **BENIGN (0)** | {metadata['class_distribution_before']['counts']['BENIGN']:,} | {metadata['class_distribution_before']['percentages']['BENIGN']}% | {metadata['class_distribution_after']['counts']['BENIGN (0)']:,} | {metadata['class_distribution_after']['percentages']['BENIGN (0)']}% |
| **PortScan (1)** | {metadata['class_distribution_before']['counts']['PortScan']:,} | {metadata['class_distribution_before']['percentages']['PortScan']}% | {metadata['class_distribution_after']['counts']['PortScan (1)']:,} | {metadata['class_distribution_after']['percentages']['PortScan (1)']}% |
| **Total** | {metadata['raw_rows']:,} | 100.0% | {metadata['cleaned_rows']:,} | 100.0% |

## Train / Test Split (seed=42)
- **Train Set (80%):** {metadata['train_rows']:,} rows
  - BENIGN (0): {metadata['train_class_counts']['BENIGN (0)']:,}
  - PortScan (1): {metadata['train_class_counts']['PortScan (1)']:,}
- **Test Set (20%):** {metadata['test_rows']:,} rows
  - BENIGN (0): {metadata['test_class_counts']['BENIGN (0)']:,}
  - PortScan (1): {metadata['test_class_counts']['PortScan (1)']:,}

## Pipeline Architecture
1. **Cleaning:** Column renaming (snake_case), infinite values replacement (`inf` -> `nan`), null elimination, deduplication, zero-variance feature pruning.
2. **Label Encoding:** `BENIGN` -> 0, `PortScan` -> 1.
3. **PySpark VectorAssembler:** Assembles {metadata['feature_count']} numerical flow features into `raw_features`.
4. **PySpark StandardScaler:** Standardizes features (`withStd=True`, `withMean=False`) into `features` vector column.
5. **Output Artifacts:**
   - Train Parquet: `{metadata['train_path']}`
   - Test Parquet: `{metadata['test_path']}`
   - Fitted Pipeline Model: `{metadata['pipeline_model_path']}`
""")
        print(f"Saved human-readable report to: {md_report_path}")

        print("\n================ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ================")

    finally:
        stop_spark_session(spark)

if __name__ == "__main__":
    run_pipeline()
