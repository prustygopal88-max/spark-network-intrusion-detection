# CICIDS2017 Preprocessing & Spark Feature Pipeline Report

## Overview
- **Raw Rows:** 286,467
- **Cleaned Rows:** 213,777
- **Duplicates Removed:** 72,353
- **Missing / Infinite Rows Dropped:** 337
- **Constant Columns Removed:** 10 (bwd_psh_flags, fwd_urg_flags, bwd_urg_flags, cwe_flag_count, fwd_avg_bytes_bulk, fwd_avg_packets_bulk, fwd_avg_bulk_rate, bwd_avg_bytes_bulk, bwd_avg_packets_bulk, bwd_avg_bulk_rate)
- **Features Retained:** 68
- **Final Feature Vector Dimension:** 68
- **Execution Time:** 126.11 seconds

## Class Distribution Before vs After

| Class | Original Count | Original % | Cleaned Count | Cleaned % |
|:---|:---:|:---:|:---:|:---:|
| **BENIGN (0)** | 127,537 | 44.52% | 123,083 | 57.58% |
| **PortScan (1)** | 158,930 | 55.48% | 90,694 | 42.42% |
| **Total** | 286,467 | 100.0% | 213,777 | 100.0% |

## Train / Test Split (seed=42)
- **Train Set (80%):** 171,195 rows
  - BENIGN (0): 98,749
  - PortScan (1): 72,446
- **Test Set (20%):** 42,582 rows
  - BENIGN (0): 24,334
  - PortScan (1): 18,248

## Pipeline Architecture
1. **Cleaning:** Column renaming (snake_case), infinite values replacement (`inf` -> `nan`), null elimination, deduplication, zero-variance feature pruning.
2. **Label Encoding:** `BENIGN` -> 0, `PortScan` -> 1.
3. **PySpark VectorAssembler:** Assembles 68 numerical flow features into `raw_features`.
4. **PySpark StandardScaler:** Standardizes features (`withStd=True`, `withMean=False`) into `features` vector column.
5. **Output Artifacts:**
   - Train Parquet: `C:\Users\prust\OneDrive\Desktop\spark-network-intrusion-detection\data\processed\train.parquet`
   - Test Parquet: `C:\Users\prust\OneDrive\Desktop\spark-network-intrusion-detection\data\processed\test.parquet`
   - Fitted Pipeline Model: `C:\Users\prust\OneDrive\Desktop\spark-network-intrusion-detection\results\models\feature_pipeline_model`
