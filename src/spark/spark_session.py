import os
import sys
from pyspark.sql import SparkSession

def setup_spark_environment():
    """Ensure JAVA_HOME, HADOOP_HOME, and Python bindings are active on Windows."""
    # Ensure JAVA_HOME
    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
        default_jdk = r"C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
        if os.path.exists(default_jdk):
            os.environ["JAVA_HOME"] = default_jdk
            os.environ["PATH"] = os.path.join(default_jdk, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Ensure HADOOP_HOME for Windows File I/O
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hadoop_dir = os.path.join(project_root, "hadoop")
    if os.path.exists(hadoop_dir):
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["PATH"] = os.path.join(hadoop_dir, "bin") + os.pathsep + os.environ.get("PATH", "")

    # Explicitly bind Python worker executable to match current interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def get_spark_session(app_name="SparkNIDS", master="local[*]"):
    """
    Creates and returns a local SparkSession configured for Windows.
    """
    setup_spark_environment()

    spark = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", "file:///C:/temp/spark-warehouse")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark

def stop_spark_session(spark):
    """Cleanly stops the active SparkSession."""
    if spark:
        try:
            spark.stop()
        except Exception:
            pass

if __name__ == "__main__":
    spark = get_spark_session("SparkEnvVerification")
    print(f"SparkSession active: version {spark.version}")
    stop_spark_session(spark)
    print("SparkSession stopped cleanly.")
