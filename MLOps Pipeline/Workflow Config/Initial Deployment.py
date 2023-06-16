# Databricks notebook source
# MAGIC %md
# MAGIC ## Installations

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install databricks && pip install databricks-feature-store
# MAGIC %pip install xgboost
# MAGIC %pip install tensorflow
# MAGIC %pip install protobuf
# MAGIC %pip install mlflow keras

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col,concat, when, lit, to_date, date_sub, max as max_, rand, lpad, concat_ws,sum,mean
from pyspark.ml.feature import VectorAssembler, VectorIndexer,OneHotEncoder, StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType, TimestampType, DateType 
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras
import mlflow.sklearn
import mlflow.models.signature as sch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import scipy.stats as stats
from xgboost import plot_importance, plot_tree, XGBRegressor
from xgboost.spark import SparkXGBRegressor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import unittest 
import requests
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

train_start = '2015-01-01' 
train_end = '2021-12-31'
test_start = '2022-01-01'
test_end = '2023-01-01'
db = 'df_dev'
feauture_store = 'hourly_forecasting_features'
consumption_countries_hourly ='final_consumption_countries_hourly'
model_name = 'pyspark_mlflow_model'
access_token = 'dapie24d3f30586ca9b17dbd6d28ce208086-2'
databricks_instance = 'adb-8855338042472349.9.azuredatabricks.net'
countries = ["belgium", "denmark", "france", "germany", "greece", "italy", "luxembourg", "netherlands", "spain", "sweden","switzerland"] #new
experiment_id_training = '3578670731332255'
experiment_id_retraining = '3578670731332164'
fs = feature_store.FeatureStoreClient()
pip_requirements = ["pyspark==3.4.0", "mlflow==2.3.2", "xgboost==1.7.5"]
user = 'filippos.priovolos01@gmail.com'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema

# COMMAND ----------

input_schema = Schema([
    ColSpec("integer", "belgium"),
    ColSpec("integer", "denmark"),
    ColSpec("integer", "france"),
    ColSpec("integer", "germany"),
    ColSpec("integer", "greece"),
    ColSpec("integer", "italy"),
    ColSpec("integer", "luxembourg"),
    ColSpec("integer", "netherlands"),
    ColSpec("integer", "spain"),
    ColSpec("integer", "sweden"),
    ColSpec("integer", "switzerland"),
    ColSpec("integer", "HOUR"),
    ColSpec("integer", "DAY_OF_WEEK"),
    ColSpec("integer", "MONTH"),
    ColSpec("integer", "QUARTER"),
    ColSpec("integer", "YEAR"),
    ColSpec("integer", "DAY_OF_YEAR"),
    ColSpec("integer", "DAY_OF_MONTH"),
    ColSpec("integer", "WEEK_OF_YEAR"),
    ColSpec("double", "ROLLING_MEAN_24H"),
    ColSpec("double", "ROLLING_STD_24H"),
    ColSpec("double", "ROLLING_SUM_7D"),
    ColSpec("double", "PREV_DAY_CONSUMPTION"),
    ColSpec("double", "PREV_WEEK_CONSUMPTION"),
    ColSpec("double", "PREVIOUS_MONTH_CONSUMPTION")
])

output_schema = Schema([ColSpec("double", "HOURLY_CONSUMPTION_MW")])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Signature

# COMMAND ----------

# Create a model signature from the input and output schemas
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
