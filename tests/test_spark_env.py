import os
import sys

# Auto-detect and set JAVA_HOME if not present in the current terminal session
if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
    candidate = r"C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
    if os.path.exists(candidate):
        os.environ["JAVA_HOME"] = candidate
        os.environ["PATH"] = os.path.join(candidate, "bin") + os.pathsep + os.environ.get("PATH", "")

# Explicitly ensure Spark worker uses the exact same Python binary
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pyspark
from pyspark.sql import SparkSession

print(f"PySpark version: {pyspark.__version__}")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")

# Initialize SparkSession with Windows-safe configurations
spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("SparkSmokeTest")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

# Create a tiny DataFrame
sample_data = [("Normal", 0), ("DDoS", 1), ("PortScan", 1)]
columns = ["traffic_label", "is_malicious"]
df = spark.createDataFrame(sample_data, columns)

# Perform simple operation
total_count = df.count()
print(f"Smoke test DataFrame row count: {total_count}")
df.show()

# Stop Spark cleanly
spark.stop()
print("SMOKE TEST STATUS: PASSED")
