# Databricks notebook source
# MAGIC %pip install tensorflow && pip install mlflow
# MAGIC %pip install databricks && pip install databricks-feature-store

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

def create_names(list_of_names, granularity):
    output_names = []
    for name in list_of_names:
        for i in range(0, granularity):
            output_names.append(name + '_' + str(i))
    return output_names

def create_training_val_test(df, val_id, test_id):
    features = df.select(F.collect_list('features')).first()[0]
    labels = df.select(F.collect_list('label')).first()[0]
    X_train = features[:val_id]
    X_val = features[val_id:test_id]
    X_test = features[test_id:]
    y_train = labels[:val_id]
    y_val = labels[val_id:test_id]
    y_test = labels[test_id:]
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

# COMMAND ----------

scenarios = ['hour', 'day', 'month']
current_scenario = scenarios[0]
db = 'df_landing'
feauture_store = 'DL_forecasting_features_' + current_scenario
fs = feature_store.FeatureStoreClient()
data = fs.read_table(name= f'{db}.{feauture_store}')
size_of_data = data.count()
user = 'nikitas.maragkos@gr.ey.com'

if current_scenario == 'hour':
    granularity = 24
    maximum_past_feature = 28 * granularity
    horizon = 24
    val_size = 365
    test_size = 365
    val_start_id = size_of_data - (test_size + val_size)
    test_start_id = size_of_data - (test_size)
elif current_scenario == 'day':
    granularity = 1
    maximum_past_feature = 7
    horizon = 1
    val_size = 365
    test_size = 365
    val_start_id = size_of_data - (test_size + val_size)
    test_start_id = size_of_data - (test_size)
elif current_scenario == 'month':
    granularity = 1
    maximum_past_feature = 3
    horizon = 1
    val_size = 12
    test_size = 12
    val_start_id = size_of_data - (test_size + val_size)
    test_start_id = size_of_data - (test_size)

# COMMAND ----------

featuresCols = data.columns
columns_to_delete = ['date', 'key', 'ID']
target_names = create_names(['y'], granularity)
for name in target_names + columns_to_delete:
    featuresCols.remove(name)

# COMMAND ----------

stages = []
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
vectorAssemblerLabel = VectorAssembler(inputCols=target_names, outputCol="label")
stages += [vectorAssembler, vectorAssemblerLabel]

# Set Pipeline
pipeline = Pipeline(stages=stages)

# Fit Pipeline to Data
pipeline_model = pipeline.fit(data)

# Transform Data using Fitted Pipeline
df_transform = pipeline_model.transform(data)

df_transform_fin = df_transform.select('features','label')

X_train, y_train, X_val, y_val, X_test, y_test = create_training_val_test(df_transform_fin, val_start_id, test_start_id)

# COMMAND ----------

def create_model():
  model = Sequential()
  model.add(Dense(600, input_dim=len(X_train[0]), activation="relu"))
  model.add(Dense(300, activation="relu"))
  model.add(Dense(100, activation="relu"))
  model.add(Dense(50, activation="relu"))
  model.add(Dense(horizon, activation="linear"))
  return model

model = create_model()
 
model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time

# In the following lines, replace <username> with your username.
experiment_log_dir = f"/dbfs/{user}/tb"
checkpoint_path = f"/dbfs/{user}/keras_checkpoint_weights.ckpt"
epochs = 1000
 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=100)

with mlflow.start_run() as run:
    mlflow.tensorflow.autolog()
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data = (X_val , y_val), epochs=epochs, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])
    end_time = time.time()
    kerasURI = run.info.artifact_uri

 
model_name = "greece_keras"
model_uri = kerasURI+"/model"
new_model_version = mlflow.register_model(model_uri, model_name)
 
# Registering the model takes a few seconds, so add a delay before continuing with the next cell
time.sleep(5)

# COMMAND ----------

keras_model = mlflow.keras.load_model(f"models:/{model_name}/{new_model_version.version}")
keras_pred = keras_model.predict(X_test)
keras_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Metrics/Parameters to be logged

# COMMAND ----------

def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100
def mae(actual, pred):
    return np.mean(np.abs(actual - pred)) 
def mse(actual, pred):
    return np.mean( (actual - pred) ** 2)
def rmse(actual, pred):
    return sqrt(mse(actual, pred))
def r2(actual, pred):
    return 

# COMMAND ----------

np.mean(np.abs( (keras_pred.flatten() - y_test.flatten()) / y_test.flatten() )) 

# COMMAND ----------

mape(y_test.flatten(),keras_pred.flatten())

# COMMAND ----------

# Metrcis
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

#Hyperparameters

# Get the stages of the pipeline
stages = pipelineModel.stages

# Iterate over the stages and retrieve their parameter values
param_dict = {}
for stage in stages:
    stage_params = stage.extractParamMap()
    param_dict.update({param.name: stage_params[param] for param in stage_params.keys()})

hyperparameters = {
    param.name: param_dict['estimatorParamMaps'][0][param]
    for param in param_dict['estimatorParamMaps'][0].keys()
}

#Model Training Time
training_time = end_time - start_time

#Model Training/Testing Data Size
training_size = train_df.count()
testing_size = test_df.count()

#Current Time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

#Description
description = "The logged model is an XGBoost regressor that has been trained to predict DAILY_CONSUMPTION_MW based on various input features. It performs well in accurately estimating energy consumption. The model takes into account important factors and patterns present in the data to make reliable predictions. It has been fine-tuned using cross-validation and optimized hyperparameters to ensure its effectiveness"

#Model Tags
tags = {
    "model_type": "XGBoost Regressor",
    "dataset": "Energy Consumption",
    "application": "Energy Management",
    "framework": "PySpark"
}

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


