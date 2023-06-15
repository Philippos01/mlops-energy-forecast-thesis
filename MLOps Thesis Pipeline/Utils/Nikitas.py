# Databricks notebook source
# MAGIC %pip install tensorflow && pip install mlflow

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
import pyspark.sql.functions as F
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns

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

# COMMAND ----------

from pyspark.sql.functions import col, sum
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def create_date_features(date):
    weekday = date.weekday()
    day = date.day
    month = date.month
    return [weekday, day, month]
    
def create_date_features_2(date):
    pd_dt = pd.to_datetime(date)
    dayofweek = pd_dt.dayofweek
    month = pd_dt.month
    quarter = pd_dt.quarter
    year = pd_dt.year
    dayofyear = pd_dt.dayofyear
    day = pd_dt.day
    weekofyear = pd_dt.weekofyear
    return [pd_dt, dayofweek, month, year, quarter, dayofyear, day, weekofyear]

def create_historical_load_features(past_load, granularity):
    previous_day_load = past_load[-granularity:]
    previous_week_load = past_load[-(7 * granularity) : -(7 * granularity) + granularity]
    previous_month_load = past_load[-(28 * granularity) : -(28 * granularity) + granularity]
    return previous_day_load + previous_week_load + previous_month_load

def create_names(list_of_names, granularity):
    output_names = []
    for name in list_of_names:
        for i in range(0, granularity):
            output_names.append(name + '_' + str(i))
    return output_names

def create_dictionary_of_lists(column_names):
    dictionary = {}
    for column in column_names:
        dictionary[column] = []
    return dictionary

def add_multiple_values_to_multiple_keys(list_of_values, key_name, dictionary):
    for key, value in zip(key_name,list_of_values):
        dictionary[key] += [value]
    return dictionary
 
def create_feature_dataset_hour( df, date_column, target_column, maximum_past_feature, granularity):
    dates = df[date_column]
    targets = df[target_column].values.tolist()
    start_id = maximum_past_feature
    final_id = len(dates)
    X = []
    y = []
    final_dates = []
    historical_data_columns = ['previous_day_load', 'previous_week_load', 'previous_month_load']
    historical_data_columns = create_names(historical_data_columns, granularity)
    date_columns = ['date', 'day_of_week', 'month', 'year', 'quarter', 'day_of_year', 'day_of_month', 'week_of_year' ]
    target_column_name = ['y']
    target_column_names = create_names(target_column_name, granularity)
    column_names = date_columns + historical_data_columns + target_column_names
    final_dictionary = create_dictionary_of_lists(column_names)
    for i in range(start_id, final_id, granularity):
        labels = targets[i:i + granularity]
        date_features = create_date_features_2(dates[i])
        historical_load_features = create_historical_load_features(targets[:i], granularity)
        final_dictionary = add_multiple_values_to_multiple_keys(date_features + historical_load_features + labels, column_names, final_dictionary)
    dataframe = pd.DataFrame(final_dictionary)
    return dataframe

# COMMAND ----------

database = 'df_landing'
table = 'greece_total_consuption_hourly' # typo needs correction
user = 'nikitas.maragkos@gr.ey.com'

# COMMAND ----------

df = spark.read.table(f"{database}.{table}")

# COMMAND ----------

date_column = 'Time'
target_column = 'Load_Actual'
granularity = 24
maximum_past_feature = 28 * granularity
new_df = create_feature_dataset_hour(df.toPandas(), date_column, target_column, maximum_past_feature, granularity)

# COMMAND ----------

sparkDF=spark.createDataFrame(new_df) 
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

featuresCols = sparkDF.columns
featuresCols.remove('date')
target_names = create_names(['y'], granularity)
for name in target_names:
    featuresCols.remove(name)

# COMMAND ----------

stages = []
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
vectorAssemblerLabel = VectorAssembler(inputCols=target_names, outputCol="label")
stages += [vectorAssembler, vectorAssemblerLabel]

# Set Pipeline
pipeline = Pipeline(stages=stages)

# Fit Pipeline to Data
pipeline_model = pipeline.fit(sparkDF)

# Transform Data using Fitted Pipeline
df_transform = pipeline_model.transform(sparkDF)

df_transform_fin = df_transform.select('features','label')
#df_transform_fin.limit(5).toPandas()

X_train, y_train, X_val, y_val, X_test, y_test = create_training_val_test(df_transform_fin, 2994 - 365 - 365, 2994 - 365)

# COMMAND ----------

def create_model():
  model = Sequential()
  model.add(Dense(300, input_dim=79, activation="relu"))
  model.add(Dense(100, activation="relu"))
  model.add(Dense(24, activation="linear"))
  return model

# COMMAND ----------

model = create_model()
 
model.compile(loss="mse",
              optimizer="Adam",
              metrics=["mse"])

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
 
# In the following lines, replace <username> with your username.
experiment_log_dir = f"/dbfs/{user}/tb"
checkpoint_path = f"/dbfs/{user}/keras_checkpoint_weights.ckpt"
 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=100)
 
history = model.fit(X_train, y_train, validation_data = (X_val , y_val), epochs=10, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

kerasURI = run.info.artifact_uri

# COMMAND ----------

import time
 
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

dbutils.tensorboard.stop()

# COMMAND ----------



    with mlflow.start_run(nested=True) as run:

        experiment = mlflow.get_experiment(experiment_id_retraining)
        if experiment:
            experiment_name = experiment.name
            mlflow.set_experiment(experiment_name)
            print(f"Active experiment set to '{experiment_name}'")
        else:
            print(f"No experiment found with name '{experiment_name}'")
        
        # Define the output schema
        output_schema = sch.Schema([sch.ColSpec("float", "DAILY_CONSUMPTION_MW")])

        # Create a model signature from the input and output schemas
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log the model input schema
        schema = {"input_schema": list(concatenated_df.columns[:-1]),"output_schema":concatenated_df.columns[-1]}
        mlflow.log_dict(schema, "schema.json")

        # Log some tags for the model
        mlflow.set_tags(tags)

        # Log some parameters for the model
        mlflow.log_dict(hyperparameters, "hyperparams.json")

        # Log the evaluation metrics as metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        #Log the time taken to train as metric
        mlflow.log_metric("Training Time(sec)", training_time)

        # Log evaluation metrics as artifact
        metrics = {"R2": r2, "MSE": mse, "RMSE": rmse, 'MAE':mae,'Training Time(sec)':training_time}
        mlflow.log_dict(metrics, "metrics.json")

        # Log the model description as artifact
        mlflow.log_text(description, "description.txt")
        
        # Log the current timestamp as the code version
        mlflow.log_param("code_version", current_time)

        # Log additional important parameters for comparison
        mlflow.log_param("n_estimators", hyperparameters["n_estimators"])
        mlflow.log_param("max_depth", hyperparameters["max_depth"])
        mlflow.log_param("learning_rate", hyperparameters["learning_rate"])
        mlflow.log_param("training_data_size", training_size)
        mlflow.log_param("testing_data_size", testing_size)

        # Log the model with its signature
        mlflow.spark.log_model(xgb_model, artifact_path="model", signature=signature,pip_requirements=pip_requirements)

        # Register the model with its signature
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="pyspark_mlflow_model")

        # Get the latest model version(The one that we now registered)
        client = MlflowClient()
        model_version = client.get_latest_versions("pyspark_mlflow_model")[0].version

        # Save your data to a new DBFS directory for each run
        data_path = f"dbfs:/FileStore/Data_Versioning/data_model_v{model_version}.parquet"
        concatenated_df.write.format("parquet").save(data_path)
 
        # Log the DBFS path as an artifact
        with open("data_path.txt", "w") as f:
            f.write(data_path)
        mlflow.log_artifact("data_path.txt")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

assemble = VectorAssembler(inputCols=imputed_col, outputCol='assembled_features', handleInvalid='error')
a_data = assemble.transform(impute_data)
scaler = MinMaxScaler(min=0.0, max=1.0, inputCol='assembled_features', outputCol='features')
s_data = scaler.fit(a_data).transform(a_data)
