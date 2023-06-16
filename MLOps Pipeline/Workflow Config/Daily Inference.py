# Databricks notebook source
# MAGIC %md
# MAGIC ## Installations

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install xgboost
# MAGIC %pip install databricks && pip install databricks-feature-store
# MAGIC #%pip install mlflow==2.4 numpy==1.22.4 protobuf==4.23.2 tensorflow==2.12.0
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import mlflow
from databricks import feature_store
from pyspark.sql.functions import col, sum, date_sub, to_date, hour,lit,add_months,date_format,expr,abs
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import matplotlib.pyplot as plt
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.mllib.evaluation import RegressionMetrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

countries=['belgium','denmark','france','germany','greece','italy','luxembourg','netherlands','spain','sweden','switzerland']
model_name = 'pyspark_mlflow_model'
db = 'df_dev'
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

spark.sql("USE df_dev")

# COMMAND ----------

from datetime import datetime, timedelta
substract_days = 162
date = (datetime.today() - timedelta(days=substract_days)).strftime('%Y-%m-%d')
yesterdate = (datetime.today() - timedelta(days=1) - timedelta(days=substract_days)).strftime('%Y-%m-%d')

# COMMAND ----------


# Check if the row exists
row_exists = spark.sql(f"""
    SELECT 1
    FROM inference_daily
    WHERE execution_date = '{date}' AND execution_yesterdate = '{yesterdate}'
""").collect()

# If row does not exist, insert it
if not row_exists:
    spark.sql(f"""
        INSERT INTO inference_daily (execution_date, execution_yesterdate)
        VALUES ('{date}', '{yesterdate}')
    """)

# COMMAND ----------

# Read the table
df = spark.table("inference_daily")

# Show the contents of the table
df.show()
