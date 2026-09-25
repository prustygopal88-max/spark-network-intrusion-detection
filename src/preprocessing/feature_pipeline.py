import os
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler

def build_spark_feature_pipeline(feature_cols, with_std=True, with_mean=False):
    """
    Builds a Spark MLlib Pipeline consisting of:
    1. VectorAssembler: Combines all individual numeric flow features into a single feature vector
    2. StandardScaler: Normalizes and scales feature distributions for linear/tree models
    """
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=with_std,
        withMean=with_mean
    )

    pipeline = Pipeline(stages=[assembler, scaler])
    return pipeline

def fit_feature_pipeline(train_df, feature_cols, save_path=None):
    """
    Fits the feature processing pipeline strictly on training data
    to prevent data leakage, and optionally saves the fitted model.
    """
    pipeline = build_spark_feature_pipeline(feature_cols)
    pipeline_model = pipeline.fit(train_df)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pipeline_model.write().overwrite().save(save_path)
        print(f"Fitted Feature PipelineModel saved to: {save_path}")

    return pipeline_model

def load_feature_pipeline(model_path):
    """Loads a previously fitted PipelineModel."""
    return PipelineModel.load(model_path)
