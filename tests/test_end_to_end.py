import os
import sys
import json
import unittest
import pandas as pd

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.spark_session import get_spark_session, stop_spark_session
from src.models.predictor import (
    load_inference_artifacts,
    predict_single_flow,
    predict_batch_flows,
    get_feature_columns
)

class TestSparkNIDSEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark_session("AutomatedTestSuite")
        cls.spark, cls.pipeline, cls.models = load_inference_artifacts()
        cls.feature_cols = get_feature_columns()
        
        presets_path = os.path.join(PROJECT_ROOT, "results", "sample_presets.json")
        with open(presets_path, "r") as f:
            cls.presets = json.load(f)

    @classmethod
    def tearDownClass(cls):
        stop_spark_session(cls.spark)

    def test_01_dataset_files_exist(self):
        """Verify all core dataset partitions exist on disk."""
        raw_csv = os.path.join(PROJECT_ROOT, "data", "raw", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        train_pq = os.path.join(PROJECT_ROOT, "data", "processed", "train.parquet")
        test_pq = os.path.join(PROJECT_ROOT, "data", "processed", "test.parquet")
        demo_csv = os.path.join(PROJECT_ROOT, "data", "processed", "demo_batch_sample.csv")

        self.assertTrue(os.path.exists(raw_csv), "Raw dataset CSV missing")
        self.assertTrue(os.path.exists(train_pq), "Processed train.parquet missing")
        self.assertTrue(os.path.exists(test_pq), "Processed test.parquet missing")
        self.assertTrue(os.path.exists(demo_csv), "Demo batch sample CSV missing")

    def test_02_model_artifacts_loaded(self):
        """Verify that all three models and the feature pipeline are loaded."""
        self.assertIsNotNone(self.pipeline, "Feature pipeline is None")
        self.assertIn("Random Forest", self.models)
        self.assertIn("Gradient-Boosted Trees", self.models)
        self.assertIn("Logistic Regression", self.models)
        self.assertEqual(len(self.feature_cols), 68, "Expected 68 feature columns")

    def test_03_single_flow_prediction_benign(self):
        """Verify single prediction on normal traffic preset."""
        res = predict_single_flow(self.presets["BENIGN"], model_name="Random Forest")
        self.assertEqual(res["prediction_label"], "BENIGN")
        self.assertGreater(res["confidence"], 0.90)
        self.assertEqual(res["class_id"], 0)

    def test_04_single_flow_prediction_portscan(self):
        """Verify single prediction on PortScan attack preset."""
        res = predict_single_flow(self.presets["PORTSCAN"], model_name="Random Forest")
        self.assertEqual(res["prediction_label"], "PORTSCAN")
        self.assertGreater(res["confidence"], 0.90)
        self.assertEqual(res["class_id"], 1)

    def test_05_batch_prediction_demo(self):
        """Verify batch prediction on 50 sample records."""
        demo_path = os.path.join(PROJECT_ROOT, "data", "processed", "demo_batch_sample.csv")
        df_demo = pd.read_csv(demo_path).head(50)
        out_df, summary = predict_batch_flows(df_demo, model_name="Random Forest")

        self.assertEqual(summary["total_analyzed"], 50)
        self.assertIn("Predicted_Class", out_df.columns)
        self.assertIn("Threat_Score (%)", out_df.columns)
        self.assertIn("Confidence (%)", out_df.columns)
        self.assertGreater(summary["portscan_count"] + summary["benign_count"], 0)

    def test_06_app_syntax_and_imports(self):
        """Verify that app/app.py can be compiled and parsed cleanly."""
        app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
        self.assertTrue(os.path.exists(app_path))
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, app_path, "exec")
        self.assertIsNotNone(compiled)

if __name__ == "__main__":
    unittest.main()
